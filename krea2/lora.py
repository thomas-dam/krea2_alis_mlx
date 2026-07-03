"""Runtime LoRA adapters for Krea2 transformer modules."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


@dataclass(frozen=True)
class LoRAReport:
    path: str
    applied: int


class LoRALinear(nn.Module):
    """Wrap an MLX Linear/QuantizedLinear with a non-destructive LoRA delta."""

    def __init__(self, base: nn.Module, down: mx.array, up: mx.array, scale: float = 1.0, alpha_scale: float = 1.0):
        super().__init__()
        self.base = base
        self.down = down
        self.up = up
        self.scale = float(scale)
        self.alpha_scale = float(alpha_scale)

    def set_scale(self, scale: float):
        self.scale = float(scale)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base(x)
        dt = x.dtype
        delta = mx.matmul(mx.matmul(x.astype(self.down.dtype), self.down.T), self.up.T)
        return out + (delta * self.scale * self.alpha_scale).astype(dt)

    def _delta_weight(self) -> mx.array:
        return mx.matmul(self.up, self.down) * self.scale * self.alpha_scale

    def fuse(self, *, requantize: bool = True) -> nn.Module:
        """Fold the LoRA delta into the wrapped Linear/QuantizedLinear weight."""
        if isinstance(self.base, nn.Linear):
            fused = nn.Linear(
                int(self.base.weight.shape[1]),
                int(self.base.weight.shape[0]),
                bias="bias" in self.base,
            )
            fused.weight = (
                self.base.weight.astype(mx.float32) + self._delta_weight().astype(mx.float32)
            ).astype(self.base.weight.dtype)
            if "bias" in self.base:
                fused.bias = self.base.bias
            return fused

        if isinstance(self.base, nn.QuantizedLinear):
            out_dims, packed_in_dims = self.base.weight.shape
            in_dims = int(packed_in_dims) * 32 // int(self.base.bits)
            base_weight = mx.dequantize(
                self.base.weight,
                self.base.scales,
                self.base.biases,
                self.base.group_size,
                self.base.bits,
                mode=self.base.mode,
            )
            fused = nn.Linear(in_dims, int(out_dims), bias="bias" in self.base)
            fused.weight = (
                base_weight.astype(mx.float32) + self._delta_weight().astype(mx.float32)
            ).astype(mx.bfloat16)
            if "bias" in self.base:
                fused.bias = self.base.bias
            if not requantize:
                return fused
            return nn.QuantizedLinear.from_linear(
                fused,
                group_size=self.base.group_size,
                bits=self.base.bits,
                mode=self.base.mode,
            )

        raise TypeError(f"Cannot fuse LoRA into {type(self.base).__name__}.")


def _module_at(root: nn.Module, path: str):
    cur = root
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _parent_and_name(root: nn.Module, path: str):
    if "." not in path:
        return root, path
    parent_path, name = path.rsplit(".", 1)
    return _module_at(root, parent_path), name


def _linear_shape(module: nn.Module) -> tuple[int, int] | None:
    if isinstance(module, LoRALinear):
        module = module.base
    if isinstance(module, nn.Linear):
        out_dims, in_dims = module.weight.shape
        return int(in_dims), int(out_dims)
    if isinstance(module, nn.QuantizedLinear):
        out_dims, packed_in_dims = module.weight.shape
        in_dims = int(packed_in_dims) * 32 // int(module.bits)
        return in_dims, int(out_dims)
    return None


def _set_module(root: nn.Module, path: str, module: nn.Module):
    parent, name = _parent_and_name(root, path)
    if isinstance(parent, list):
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def _load_lora(path: str):
    raw = mx.load(path)
    modules: dict[str, dict[str, mx.array]] = {}
    for key, value in raw.items():
        name = key.removeprefix("diffusion_model.")
        if name.endswith(".lora_A.weight"):
            modules.setdefault(name[: -len(".lora_A.weight")], {})["down"] = value
        elif name.endswith(".lora_B.weight"):
            modules.setdefault(name[: -len(".lora_B.weight")], {})["up"] = value
        elif name.endswith(".alpha"):
            modules.setdefault(name[: -len(".alpha")], {})["alpha"] = value
    return modules


def apply_lora(model: nn.Module, path: str, scale: float = 1.0) -> LoRAReport:
    """Apply a Krea2 LoRA safetensors file to matching Linear modules.

    Expected keys are `diffusion_model.<module>.lora_A.weight` and
    `diffusion_model.<module>.lora_B.weight`. The `diffusion_model.` prefix is optional.
    """
    modules = _load_lora(path)
    if not modules:
        raise ValueError(f"No LoRA tensors found in {path!r}.")

    missing = []
    incompatible = []
    applied = 0
    for module_path, tensors in sorted(modules.items()):
        down = tensors.get("down")
        up = tensors.get("up")
        if down is None or up is None:
            missing.append(module_path)
            continue
        try:
            target = _module_at(model, module_path)
        except (AttributeError, IndexError, KeyError, ValueError):
            incompatible.append(f"{module_path}: no matching module")
            continue
        shape = _linear_shape(target)
        if shape is None:
            incompatible.append(f"{module_path}: target is not Linear/QuantizedLinear")
            continue
        in_dims, out_dims = shape
        if tuple(down.shape) != (down.shape[0], in_dims) or tuple(up.shape) != (out_dims, down.shape[0]):
            incompatible.append(
                f"{module_path}: LoRA {tuple(down.shape)} + {tuple(up.shape)} "
                f"does not match Linear {in_dims}->{out_dims}"
            )
            continue
        alpha = tensors.get("alpha")
        alpha_scale = 1.0
        if alpha is not None:
            alpha_scale = float(alpha.item()) / int(down.shape[0])
        if isinstance(target, LoRALinear):
            target.down = down.astype(target.down.dtype)
            target.up = up.astype(target.up.dtype)
            target.alpha_scale = alpha_scale
            target.set_scale(scale)
        else:
            wrapped = LoRALinear(target, down.astype(mx.bfloat16), up.astype(mx.bfloat16), scale=scale, alpha_scale=alpha_scale)
            _set_module(model, module_path, wrapped)
        applied += 1

    if missing:
        raise ValueError(f"LoRA file has incomplete A/B pairs for {len(missing)} module(s): {missing[:5]}")
    if incompatible:
        sample = "; ".join(incompatible[:5])
        raise ValueError(f"LoRA is incompatible with this Krea2 transformer: {sample}")
    mx.eval(model.parameters())
    return LoRAReport(path=path, applied=applied)


def set_lora_scale(module: nn.Module, scale: float) -> int:
    """Update all active LoRA wrapper scales under a module."""
    count = 0
    for _, child in module.named_modules():
        if isinstance(child, LoRALinear):
            child.set_scale(scale)
            count += 1
    return count


def fuse_lora(module: nn.Module, *, requantize: bool = True) -> int:
    """Replace every LoRALinear under module with a fused Linear/QuantizedLinear."""
    fused = 0
    for path, child in list(module.named_modules()):
        if path and isinstance(child, LoRALinear):
            _set_module(module, path, child.fuse(requantize=requantize))
            fused += 1
    if fused:
        mx.eval(module.parameters())
    return fused
