#!/usr/bin/env python3
"""Check Krea2 LoRA compatibility before copying files into loras/."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent


LORA_SUFFIXES = (
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".alpha", "alpha"),
    (".A", "down"),
    (".B", "up"),
)


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class CompatibilityReport:
    path: Path
    compatible: bool
    kind: str
    applied: int
    reason: str


def krea2_linear_shapes() -> dict[str, tuple[int, int]]:
    """Return expected Linear input/output dimensions for this MLX Krea2 port."""
    features = 6144
    txtdim = 2560
    txtlayers = 12
    layers = 28
    channels = 16
    patch = 2
    heads = 48
    kvheads = 12
    txtheads = 20
    txtkvheads = 20

    headdim = features // heads
    kvdim = headdim * kvheads
    txt_headdim = txtdim // txtheads
    txt_kvdim = txt_headdim * txtkvheads
    mlpdim = 128 * (((int(2 * features / 3) * 4) + 127) // 128)
    txt_mlpdim = 128 * (((int(2 * txtdim / 3) * 4) + 127) // 128)

    shapes: dict[str, tuple[int, int]] = {
        "first": (channels * patch * patch, features),
        "tmlp.0": (256, features),
        "tmlp.2": (features, features),
        "txtfusion.projector": (txtlayers, 1),
        "txtmlp.1": (txtdim, features),
        "txtmlp.3": (features, features),
        "last.linear": (features, channels * patch * patch),
        "tproj.1": (features, features * 6),
    }

    for i in range(layers):
        prefix = f"blocks.{i}"
        shapes[f"{prefix}.attn.wq"] = (features, features)
        shapes[f"{prefix}.attn.wk"] = (features, kvdim)
        shapes[f"{prefix}.attn.wv"] = (features, kvdim)
        shapes[f"{prefix}.attn.gate"] = (features, features)
        shapes[f"{prefix}.attn.wo"] = (features, features)
        shapes[f"{prefix}.mlp.gate"] = (features, mlpdim)
        shapes[f"{prefix}.mlp.up"] = (features, mlpdim)
        shapes[f"{prefix}.mlp.down"] = (mlpdim, features)

    for group in ("layerwise_blocks", "refiner_blocks"):
        for i in range(2):
            prefix = f"txtfusion.{group}.{i}"
            shapes[f"{prefix}.attn.wq"] = (txtdim, txtdim)
            shapes[f"{prefix}.attn.wk"] = (txtdim, txt_kvdim)
            shapes[f"{prefix}.attn.wv"] = (txtdim, txt_kvdim)
            shapes[f"{prefix}.attn.gate"] = (txtdim, txtdim)
            shapes[f"{prefix}.attn.wo"] = (txtdim, txtdim)
            shapes[f"{prefix}.mlp.gate"] = (txtdim, txt_mlpdim)
            shapes[f"{prefix}.mlp.up"] = (txtdim, txt_mlpdim)
            shapes[f"{prefix}.mlp.down"] = (txt_mlpdim, txtdim)

    return shapes


KREA2_LINEAR_SHAPES = krea2_linear_shapes()


def normalize_module_path(path: str) -> str:
    """Map known Krea2 LoRA exporter names to this MLX module tree."""
    path = path.removeprefix("diffusion_model.")
    if path.startswith("transformer.text_fusion."):
        return "txtfusion." + path.removeprefix("transformer.text_fusion.")
    if path.startswith("text_fusion."):
        return "txtfusion." + path.removeprefix("text_fusion.")
    return path


def read_lora_header(path: Path) -> tuple[dict[str, dict[str, TensorInfo]], dict[str, TensorInfo]]:
    modules: dict[str, dict[str, TensorInfo]] = {}
    full_tensors: dict[str, TensorInfo] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            name = key.removeprefix("diffusion_model.")
            tensor = handle.get_slice(key)
            info = TensorInfo(shape=tuple(tensor.get_shape()), dtype=str(tensor.get_dtype()))
            if name in {"first.weight", "first.bias"}:
                full_tensors[name] = info
                continue
            for suffix, field in LORA_SUFFIXES:
                if name.endswith(suffix):
                    module_path = normalize_module_path(name[: -len(suffix)])
                    modules.setdefault(module_path, {})[field] = info
                    break
    return modules, full_tensors


def validate_lora_modules(modules: dict[str, dict[str, TensorInfo]]) -> tuple[int, list[str], list[str]]:
    missing: list[str] = []
    incompatible: list[str] = []
    applied = 0
    for module_path, tensors in sorted(modules.items()):
        down = tensors.get("down")
        up = tensors.get("up")
        if down is None or up is None:
            missing.append(module_path)
            continue
        shape = KREA2_LINEAR_SHAPES.get(module_path)
        if shape is None:
            incompatible.append(f"{module_path}: no matching module")
            continue
        in_dims, out_dims = shape
        rank = down.shape[0] if down.shape else 0
        if down.shape != (rank, in_dims) or up.shape != (out_dims, rank):
            incompatible.append(
                f"{module_path}: LoRA {down.shape} + {up.shape} does not match Linear {in_dims}->{out_dims}"
            )
            continue
        applied += 1
    return applied, missing, incompatible


def check_regular_lora(path: Path) -> CompatibilityReport:
    try:
        modules, _ = read_lora_header(path)
    except Exception as exc:
        return CompatibilityReport(path, False, "regular", 0, f"read error: {exc}")
    if not modules:
        return CompatibilityReport(path, False, "regular", 0, "no LoRA A/B tensors found")
    applied, missing, incompatible = validate_lora_modules(modules)
    if missing:
        return CompatibilityReport(path, False, "regular", applied, f"incomplete A/B pairs: {missing[:5]}")
    if incompatible:
        return CompatibilityReport(path, False, "regular", applied, "; ".join(incompatible[:5]))
    return CompatibilityReport(path, True, "regular", applied, f"compatible regular LoRA ({applied} module(s))")


def check_depth_lora(path: Path) -> CompatibilityReport:
    try:
        modules, full_tensors = read_lora_header(path)
    except Exception as exc:
        return CompatibilityReport(path, False, "depth", 0, f"read error: {exc}")

    first_shape = KREA2_LINEAR_SHAPES.get("first")
    if first_shape is None:
        return CompatibilityReport(path, False, "depth", 0, "transformer first projection is not linear")
    in_dims, out_dims = first_shape

    applied = 0
    first_weight = full_tensors.get("first.weight")
    first_bias = full_tensors.get("first.bias")
    if first_weight is not None:
        if first_weight.shape != (out_dims, in_dims * 2):
            return CompatibilityReport(
                path,
                False,
                "depth",
                0,
                f"first.weight {first_weight.shape} does not match expected {(out_dims, in_dims * 2)}",
            )
        if first_bias is not None and first_bias.shape != (out_dims,):
            return CompatibilityReport(path, False, "depth", 0, f"first.bias {first_bias.shape} does not match expected {(out_dims,)}")
        applied += 1
    else:
        first = modules.pop("first", None)
        if first is None or first.get("down") is None or first.get("up") is None:
            return CompatibilityReport(path, False, "depth", 0, "missing expanded first-projection tensors")
        down = first["down"]
        up = first["up"]
        rank = down.shape[0] if down.shape else 0
        if down.shape != (rank, in_dims * 2) or up.shape != (out_dims, rank):
            return CompatibilityReport(
                path,
                False,
                "depth",
                0,
                f"first LoRA {down.shape} + {up.shape} does not match expected {(rank, in_dims * 2)} + {(out_dims, rank)}",
            )
        applied += 1

    modules.pop("first", None)
    if modules:
        module_applied, missing, incompatible = validate_lora_modules(modules)
        applied += module_applied
        if missing:
            return CompatibilityReport(path, False, "depth", applied, f"incomplete A/B pairs: {missing[:5]}")
        if incompatible:
            return CompatibilityReport(path, False, "depth", applied, "; ".join(incompatible[:5]))
    if applied <= 1:
        return CompatibilityReport(path, False, "depth", applied, "depth adapter has no block LoRA tensors")
    return CompatibilityReport(path, True, "depth", applied, f"compatible depth-control LoRA ({applied} module(s))")


def check_lora(path: Path, kind: str = "regular") -> CompatibilityReport:
    if kind == "regular":
        return check_regular_lora(path)
    if kind == "depth":
        return check_depth_lora(path)
    regular = check_regular_lora(path)
    if regular.compatible:
        return regular
    depth = check_depth_lora(path)
    if depth.compatible:
        return depth
    return CompatibilityReport(path, False, "any", 0, f"regular: {regular.reason}; depth: {depth.reason}")


def iter_safetensors(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix == ".safetensors" else []
    pattern = "**/*.safetensors" if recursive else "*.safetensors"
    return sorted(input_path.glob(pattern), key=lambda p: str(p).lower())


def copy_compatible(report: CompatibilityReport, destination: Path, overwrite: bool = False) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / report.path.name
    if report.path.resolve() == target.resolve():
        return "already in destination"
    if target.exists() and not overwrite:
        return "skipped copy; destination exists"
    shutil.copy2(report.path, target)
    return f"copied to {target}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Krea2 LoRA compatibility before copying files into loras/.")
    parser.add_argument("input", type=Path, help="A .safetensors file or a folder containing .safetensors files.")
    parser.add_argument("--copy-compatible", action="store_true", help="Copy compatible files into the destination folder.")
    parser.add_argument("--dest", type=Path, default=ROOT / "loras", help="Destination folder for --copy-compatible.")
    parser.add_argument("--kind", choices=("regular", "depth", "any"), default="regular", help="Compatibility target to validate.")
    parser.add_argument("--recursive", action="store_true", help="Scan folders recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files when copying.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    files = iter_safetensors(args.input.expanduser(), recursive=args.recursive)
    if not files:
        print(f"No .safetensors files found: {args.input}", file=sys.stderr)
        return 2

    ok = 0
    bad = 0
    for path in files:
        report = check_lora(path, kind=args.kind)
        label = "OK" if report.compatible else "NO"
        print(f"{label}  {path.name}  [{report.kind}] {report.reason}")
        if report.compatible:
            ok += 1
            if args.copy_compatible:
                print(f"    {copy_compatible(report, args.dest.expanduser(), overwrite=args.overwrite)}")
        else:
            bad += 1

    print(f"\nSummary: {ok} compatible, {bad} rejected")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
