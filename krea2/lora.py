"""Runtime LoRA adapters for Krea2 transformer modules."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


@dataclass(frozen=True)
class LoRAReport:
    path: str
    applied: int


class DepthControlFirst(nn.Module):
    """Depth-control adapter for the expanded first projection.

    The base transformer was trained with image patch tokens only. Depth-control
    LoRAs add a low-rank delta whose input is [image_patch, depth_patch].
    """

    def __init__(self, base: nn.Module, down: mx.array, up: mx.array, scale: float = 1.0, alpha_scale: float = 1.0):
        super().__init__()
        self.base = base
        self.down = down
        self.up = up
        self.scale = float(scale)
        self.alpha_scale = float(alpha_scale)
        self.supports_depth_control = True

    def __call__(self, img: mx.array, control_img: mx.array | None = None, control_strength: float = 1.0) -> mx.array:
        out = self.base(img)
        if control_img is None:
            return out
        if img.shape != control_img.shape:
            raise ValueError(f"control tokens shape {control_img.shape} does not match image tokens {img.shape}.")
        x = mx.concatenate([img, control_img * float(control_strength)], axis=-1)
        dt = out.dtype
        delta = mx.matmul(mx.matmul(x.astype(self.down.dtype), self.down.T), self.up.T)
        return out + (delta * self.scale * self.alpha_scale).astype(dt)


class ExpandedDepthInput(nn.Module):
    """Full expanded input projection from Patil/Krea-2-depth-controlnet."""

    def __init__(self, base: nn.Module, weight: mx.array, bias: mx.array | None = None):
        super().__init__()
        self.base = base
        self.weight = weight
        if bias is not None:
            self.bias = bias
        self.supports_depth_control = True

    def __call__(self, img: mx.array, control_img: mx.array | None = None, control_strength: float = 1.0) -> mx.array:
        if control_img is None:
            return self.base(img)
        if img.shape != control_img.shape:
            raise ValueError(f"control tokens shape {control_img.shape} does not match image tokens {img.shape}.")
        x = mx.concatenate([img, control_img * float(control_strength)], axis=-1)
        out = mx.matmul(x.astype(self.weight.dtype), self.weight.T)
        if "bias" in self:
            out = out + self.bias
        return out.astype(img.dtype)


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
    while isinstance(module, (LoRALinear, DepthControlFirst, ExpandedDepthInput)):
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


def _normalize_lora_module_path(path: str) -> str:
    """Map known Krea2 LoRA exporter names to this MLX module tree."""
    path = path.removeprefix("diffusion_model.")
    if path.startswith("transformer.text_fusion."):
        return "txtfusion." + path.removeprefix("transformer.text_fusion.")
    if path.startswith("text_fusion."):
        return "txtfusion." + path.removeprefix("text_fusion.")
    return path


def _load_lora(path: str):
    raw = mx.load(path)
    modules: dict[str, dict[str, mx.array]] = {}
    for key, value in raw.items():
        name = _normalize_lora_module_path(key)
        if name.endswith(".lora_A.weight"):
            modules.setdefault(name[: -len(".lora_A.weight")], {})["down"] = value
        elif name.endswith(".lora_B.weight"):
            modules.setdefault(name[: -len(".lora_B.weight")], {})["up"] = value
        elif name.endswith(".alpha"):
            modules.setdefault(name[: -len(".alpha")], {})["alpha"] = value
        elif name.endswith(".A"):
            modules.setdefault(name[: -len(".A")], {})["down"] = value
        elif name.endswith(".B"):
            modules.setdefault(name[: -len(".B")], {})["up"] = value
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
        elif isinstance(target, (DepthControlFirst, ExpandedDepthInput)):
            target.base = LoRALinear(target.base, down.astype(mx.bfloat16), up.astype(mx.bfloat16), scale=scale, alpha_scale=alpha_scale)
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


def apply_depth_lora(model: nn.Module, path: str, scale: float = 1.0) -> LoRAReport:
    """Apply a depth-control LoRA with an expanded first projection.

    A valid depth-control adapter must include `first.lora_A/B` tensors whose A
    input dimension is twice the base first-projection input dimension.
    Remaining tensors are applied like ordinary LoRA modules.
    """
    raw = mx.load(path)
    modules = _load_lora(path)
    if not modules and "first.weight" not in raw and "diffusion_model.first.weight" not in raw:
        raise ValueError(f"No depth-control tensors found in {path!r}.")

    target = _module_at(model, "first")
    shape = _linear_shape(target)
    if shape is None:
        raise ValueError("Transformer first projection is not Linear/QuantizedLinear-compatible.")
    in_dims, out_dims = shape

    full_weight = raw.get("first.weight")
    if full_weight is None:
        full_weight = raw.get("diffusion_model.first.weight")
    full_bias = raw.get("first.bias")
    if full_bias is None:
        full_bias = raw.get("diffusion_model.first.bias")

    applied = 0
    if full_weight is not None:
        if tuple(full_weight.shape) != (out_dims, in_dims * 2):
            raise ValueError(
                f"Depth-control adapter {path!r} has first.weight {tuple(full_weight.shape)}, "
                f"expected ({out_dims}, {in_dims * 2})."
            )
        if full_bias is not None and tuple(full_bias.shape) != (out_dims,):
            raise ValueError(f"Depth-control adapter {path!r} has first.bias {tuple(full_bias.shape)}, expected ({out_dims},).")
        if isinstance(target, ExpandedDepthInput):
            target.weight = full_weight.astype(target.weight.dtype)
            if full_bias is not None:
                target.bias = full_bias.astype(target.weight.dtype)
        else:
            _set_module(model, "first", ExpandedDepthInput(target, full_weight.astype(mx.bfloat16), None if full_bias is None else full_bias.astype(mx.bfloat16)))
        applied += 1
    else:
        first = modules.pop("first", None)
        if first is None or first.get("down") is None or first.get("up") is None:
            raise ValueError(f"Depth-control adapter {path!r} does not contain expanded first-projection tensors.")

        down = first["down"]
        up = first["up"]
        if tuple(down.shape) != (down.shape[0], in_dims * 2) or tuple(up.shape) != (out_dims, down.shape[0]):
            raise ValueError(
                f"Depth-control adapter {path!r} has first projection LoRA {tuple(down.shape)} + {tuple(up.shape)}, "
                f"expected ({down.shape[0]}, {in_dims * 2}) + ({out_dims}, {down.shape[0]})."
            )
        alpha = first.get("alpha")
        alpha_scale = float(alpha.item()) / int(down.shape[0]) if alpha is not None else 1.0
        if isinstance(target, DepthControlFirst):
            target.down = down.astype(target.down.dtype)
            target.up = up.astype(target.up.dtype)
            target.scale = float(scale)
            target.alpha_scale = alpha_scale
        else:
            _set_module(
                model,
                "first",
                DepthControlFirst(target, down.astype(mx.bfloat16), up.astype(mx.bfloat16), scale=scale, alpha_scale=alpha_scale),
            )
        applied += 1

    if "first" in modules:
        modules.pop("first")

    if modules:
        # Apply LoRA deltas for all non-expanded modules. Patil/Krea-2-depth-controlnet
        # uses `.A`/`.B`; generic LoRAs use `.lora_A.weight`/`.lora_B.weight`.
        missing = []
        incompatible = []
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
            alpha_scale = float(alpha.item()) / int(down.shape[0]) if alpha is not None else 1.0
            if isinstance(target, LoRALinear):
                target.down = down.astype(target.down.dtype)
                target.up = up.astype(target.up.dtype)
                target.alpha_scale = alpha_scale
                target.set_scale(scale)
            elif isinstance(target, (DepthControlFirst, ExpandedDepthInput)):
                target.base = LoRALinear(target.base, down.astype(mx.bfloat16), up.astype(mx.bfloat16), scale=scale, alpha_scale=alpha_scale)
            else:
                _set_module(model, module_path, LoRALinear(target, down.astype(mx.bfloat16), up.astype(mx.bfloat16), scale=scale, alpha_scale=alpha_scale))
            applied += 1
        if missing:
            raise ValueError(f"Depth LoRA file has incomplete A/B pairs for {len(missing)} module(s): {missing[:5]}")
        if incompatible:
            sample = "; ".join(incompatible[:5])
            raise ValueError(f"Depth LoRA is incompatible with this Krea2 transformer: {sample}")

    if applied <= 1:
        raise ValueError(
            f"Depth-control adapter {path!r} loaded the expanded input projection but no block LoRA tensors. "
            "Expected rank-64 block adapters as well."
        )

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
