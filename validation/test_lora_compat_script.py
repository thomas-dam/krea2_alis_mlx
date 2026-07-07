"""Weights-free checks for the standalone LoRA compatibility gate."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import scripts/krea2

from scripts.check_lora_compat import check_lora


TMP = Path("/tmp/krea2_lora_compat_script_test")


def tensor(shape: tuple[int, ...], dtype=np.float32):
    return np.zeros(shape, dtype=dtype)


def save(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return path


def test_regular_krea2_lora_is_compatible():
    path = save(
        TMP / "regular.safetensors",
        {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight": tensor((2, 6144)),
            "diffusion_model.blocks.0.attn.wq.lora_B.weight": tensor((6144, 2)),
        },
    )
    report = check_lora(path, kind="regular")
    assert report.compatible, report.reason
    assert report.kind == "regular"
    assert report.applied == 1


def test_text_fusion_projector_alias_is_compatible():
    path = save(
        TMP / "projector_alias.safetensors",
        {
            "transformer.text_fusion.projector.lora_A.weight": tensor((1, 12)),
            "transformer.text_fusion.projector.lora_B.weight": tensor((1, 1)),
        },
    )
    report = check_lora(path, kind="regular")
    assert report.compatible, report.reason
    assert report.applied == 1


def test_non_lora_safetensors_is_rejected():
    path = save(
        TMP / "vae_like.safetensors",
        {
            "decoder.conv1.weight": tensor((2, 2)),
        },
    )
    report = check_lora(path, kind="regular")
    assert not report.compatible
    assert "no LoRA" in report.reason


def test_unknown_target_is_rejected():
    path = save(
        TMP / "unknown.safetensors",
        {
            "some.other.module.lora_A.weight": tensor((2, 4)),
            "some.other.module.lora_B.weight": tensor((8, 2)),
        },
    )
    report = check_lora(path, kind="regular")
    assert not report.compatible
    assert "no matching module" in report.reason


def test_depth_control_lora_is_compatible_for_depth_kind():
    path = save(
        TMP / "depth.safetensors",
        {
            "first.weight": tensor((6144, 128)),
            "first.bias": tensor((6144,)),
            "blocks.0.attn.wq.A": tensor((2, 6144)),
            "blocks.0.attn.wq.B": tensor((6144, 2)),
        },
    )
    depth = check_lora(path, kind="depth")
    assert depth.compatible, depth.reason
    assert depth.kind == "depth"
    assert depth.applied == 2


def main() -> int:
    test_regular_krea2_lora_is_compatible()
    test_text_fusion_projector_alias_is_compatible()
    test_non_lora_safetensors_is_rejected()
    test_unknown_target_is_rejected()
    test_depth_control_lora_is_compatible_for_depth_kind()
    print("OK: standalone LoRA compatibility checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
