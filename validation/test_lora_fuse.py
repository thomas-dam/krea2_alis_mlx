"""Fast LoRA fuse equivalence checks for plain and quantized MLX linears."""

import os
import sys

import mlx.core as mx
from mlx import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import krea2

from krea2.lora import LoRALinear, fuse_lora


def _max_abs(a, b) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def test_linear_fuse_matches_runtime_lora():
    mx.random.seed(7)
    base = nn.Linear(64, 32)
    down = mx.random.normal((4, 64)).astype(mx.bfloat16) * 0.01
    up = mx.random.normal((32, 4)).astype(mx.bfloat16) * 0.01
    x = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)

    wrapped = LoRALinear(base, down, up, scale=0.7, alpha_scale=0.5)
    fused = wrapped.fuse()
    wrapped_out = wrapped(x)
    fused_out = fused(x)
    mx.eval(wrapped_out, fused_out)

    assert isinstance(fused, nn.Linear)
    assert _max_abs(wrapped_out, fused_out) <= 0.001


def test_quantized_fuse_matches_runtime_lora_without_requantizing():
    mx.random.seed(11)
    base = nn.Linear(64, 32)
    qbase = nn.QuantizedLinear.from_linear(base, group_size=64, bits=8)
    down = mx.random.normal((4, 64)).astype(mx.bfloat16) * 0.01
    up = mx.random.normal((32, 4)).astype(mx.bfloat16) * 0.01
    x = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)

    wrapped = LoRALinear(qbase, down, up, scale=0.7, alpha_scale=0.5)
    fused = wrapped.fuse(requantize=False)
    wrapped_out = wrapped(x)
    fused_out = fused(x)
    mx.eval(wrapped_out, fused_out)

    assert isinstance(fused, nn.Linear)
    assert _max_abs(wrapped_out, fused_out) <= 0.005


def test_quantized_fuse_requantizes_to_quantized_linear():
    mx.random.seed(13)
    base = nn.Linear(64, 32)
    qbase = nn.QuantizedLinear.from_linear(base, group_size=64, bits=8)
    down = mx.random.normal((4, 64)).astype(mx.bfloat16) * 0.01
    up = mx.random.normal((32, 4)).astype(mx.bfloat16) * 0.01
    x = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)

    wrapped = LoRALinear(qbase, down, up, scale=0.7, alpha_scale=0.5)
    fused = wrapped.fuse(requantize=True)
    wrapped_out = wrapped(x)
    fused_out = fused(x)
    mx.eval(wrapped_out, fused_out)

    assert isinstance(fused, nn.QuantizedLinear)
    assert fused.group_size == qbase.group_size
    assert fused.bits == qbase.bits
    assert fused.mode == qbase.mode
    assert _max_abs(wrapped_out, fused_out) <= 0.02


def test_fuse_lora_replaces_wrappers_in_module_tree():
    mx.random.seed(17)
    model = nn.Sequential(
        LoRALinear(
            nn.Linear(64, 32),
            mx.random.normal((4, 64)).astype(mx.bfloat16) * 0.01,
            mx.random.normal((32, 4)).astype(mx.bfloat16) * 0.01,
        )
    )

    assert any(isinstance(module, LoRALinear) for _, module in model.named_modules())
    assert fuse_lora(model) == 1
    assert not any(isinstance(module, LoRALinear) for _, module in model.named_modules())


def main() -> int:
    test_linear_fuse_matches_runtime_lora()
    test_quantized_fuse_matches_runtime_lora_without_requantizing()
    test_quantized_fuse_requantizes_to_quantized_linear()
    test_fuse_lora_replaces_wrappers_in_module_tree()
    print("OK: LoRA fuse matches runtime adapters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
