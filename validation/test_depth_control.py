"""Weights-free depth-control checks for wrapper and sampler validation."""

import os
import sys

import mlx.core as mx
from mlx import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import krea2

from krea2.lora import DepthControlFirst, ExpandedDepthInput, apply_depth_lora
from krea2.sampling import sample


def test_depth_first_no_control_matches_base():
    mx.random.seed(101)
    base = nn.Linear(64, 32)
    down = mx.random.normal((4, 128)).astype(mx.bfloat16) * 0.01
    up = mx.random.normal((32, 4)).astype(mx.bfloat16) * 0.01
    wrapped = DepthControlFirst(base, down, up)
    x = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)

    a = base(x)
    b = wrapped(x)
    mx.eval(a, b)
    assert float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))) == 0.0


def test_depth_first_rejects_mismatched_tokens():
    base = nn.Linear(64, 32)
    wrapped = DepthControlFirst(
        base,
        mx.zeros((4, 128)).astype(mx.bfloat16),
        mx.zeros((32, 4)).astype(mx.bfloat16),
    )
    try:
        wrapped(mx.zeros((1, 4, 64)), mx.zeros((1, 5, 64)))
    except ValueError as e:
        assert "control tokens shape" in str(e)
    else:
        raise AssertionError("expected mismatched control token shapes to fail")


class _FakeVAE:
    spatial_scale = 8
    latent_channels = 16

    def decode(self, latent):
        b, _, h, w = latent.shape
        return mx.zeros((b, 3, 1, h * 8, w * 8))


class _FakeEncoder:
    def __call__(self, prompts):
        n = len(prompts)
        return mx.zeros((n, 2, 1, 4)), mx.ones((n, 2))


class _FakeTransformer:
    class _Cfg:
        patch = 2

    cfg = _Cfg()

    def __call__(self, img, context, t, pos, mask, control_img=None, control_strength=1.0):
        return mx.zeros_like(img)


def test_sample_rejects_control_shape_mismatch():
    try:
        sample(
            _FakeTransformer(),
            _FakeVAE(),
            _FakeEncoder(),
            ["prompt"],
            width=256,
            height=256,
            steps=2,
            control_latent=mx.zeros((1, 16, 16, 16)),
        )
    except ValueError as e:
        assert "control_latent shape" in str(e)
    else:
        raise AssertionError("expected mismatched control latent shape to fail")


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(8, 8)


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _TinyAttention()


class _TinyDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 8)
        self.blocks = [_TinyBlock()]


def test_apply_depth_lora_accepts_patil_checkpoint_keys():
    model = _TinyDepthModel()
    path = "/tmp/krea2_depth_control_test.safetensors"
    mx.save_safetensors(
        path,
        {
            "first.weight": mx.zeros((8, 8)).astype(mx.bfloat16),
            "first.bias": mx.zeros((8,)).astype(mx.bfloat16),
            "blocks.0.attn.wq.A": mx.zeros((2, 8)).astype(mx.bfloat16),
            "blocks.0.attn.wq.B": mx.zeros((8, 2)).astype(mx.bfloat16),
        },
    )

    report = apply_depth_lora(model, path)
    assert report.applied == 2
    assert isinstance(model.first, ExpandedDepthInput)


def main() -> int:
    test_depth_first_no_control_matches_base()
    test_depth_first_rejects_mismatched_tokens()
    test_sample_rejects_control_shape_mismatch()
    test_apply_depth_lora_accepts_patil_checkpoint_keys()
    print("OK: depth-control wrapper and sampler validation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
