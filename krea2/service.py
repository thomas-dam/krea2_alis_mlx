"""Shared generation service for Gradio and FastAPI entrypoints."""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL.Image import Image

from .pipeline import Krea2Pipeline, resolve_weights


log = logging.getLogger("krea2.service")
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
ASPECT_RATIOS = [
    ("1:1 · 1024×1024", "1:1"),
    ("4:3 · 1152×864", "4:3"),
    ("3:2 · 1248×832", "3:2"),
    ("16:9 · 1360×768", "16:9"),
    ("2.35:1 · 1584×672", "2.35:1"),
    ("4:5 · 896×1120", "4:5"),
    ("2:3 · 832×1248", "2:3"),
    ("9:16 · 768×1360", "9:16"),
]
ASPECT_DIMS = {
    "1:1": (1024, 1024),
    "4:3": (1152, 864),
    "3:2": (1248, 832),
    "16:9": (1360, 768),
    "2.35:1": (1584, 672),
    "4:5": (896, 1120),
    "2:3": (832, 1248),
    "9:16": (768, 1360),
}
MODELS = [("8-bit · best quality (14 GB)", "8bit"), ("mixed-4/8 · smaller (9.8 GB)", "mixed-4-8")]

_PIPE = None
_PIPE_PREC = None
_PIPE_LORA = None
_PIPE_LOCK = threading.Lock()


@dataclass
class GenerationResult:
    job_id: str
    images: list[Image]
    saved: list[str]
    metadata_path: str
    timings: dict
    width: int
    height: int


def default_precision():
    precision, _ = resolve_weights(str(ROOT), download=False)
    return precision


def normalize_lora_path(lora_path: str | None) -> str | None:
    return os.path.expanduser(lora_path.strip()) if lora_path and lora_path.strip() else None


def get_pipeline(precision: str, lora_path: str | None = None) -> Krea2Pipeline:
    """Return a cached pipeline for precision + LoRA path."""
    global _PIPE, _PIPE_PREC, _PIPE_LORA
    lora_path = normalize_lora_path(lora_path)
    if lora_path and not os.path.exists(lora_path):
        raise ValueError(f"LoRA file not found: {lora_path}")
    if _PIPE is None or _PIPE_PREC != precision or _PIPE_LORA != lora_path:
        prec, path = resolve_weights(str(ROOT), precision=precision, download=True)
        _PIPE, _PIPE_PREC, _PIPE_LORA = None, None, None
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()
        _PIPE = Krea2Pipeline(
            path,
            precision=prec,
            base_dir=os.environ.get("KREA2_BASE_DIR"),
            lora_path=lora_path,
        )
        _PIPE_PREC = prec
        _PIPE_LORA = lora_path
    return _PIPE


def save_outputs(
    images: list[Image],
    *,
    prompt: str,
    model: str,
    lora_path: str | None,
    lora_strength: float,
    aspect_ratio: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    safety_on: bool,
    timings: dict,
):
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    metadata = {
        "prompt": prompt,
        "model": model,
        "lora_path": lora_path or None,
        "lora_strength": float(lora_strength),
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "steps": int(steps),
        "seed": int(seed),
        "num_images": len(images),
        "safety_on": bool(safety_on),
        "timings": timings,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "images": [],
    }
    for i, image in enumerate(images, start=1):
        path = OUTPUT_DIR / f"{stamp}_{i:02d}.png"
        image.save(path)
        saved.append(str(path))
        metadata["images"].append(path.name)
    if "save_started_at" in timings:
        timings["save_seconds"] = round(time.perf_counter() - timings.pop("save_started_at"), 3)
    if "run_started_at" in timings:
        timings["total_seconds"] = round(time.perf_counter() - timings.pop("run_started_at"), 3)
    meta_path = OUTPUT_DIR / f"{stamp}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return saved, str(meta_path)


def generate_and_save(
    *,
    prompt: str,
    model: str = "8bit",
    lora_path: str | None = None,
    lora_strength: float = 1.0,
    aspect_ratio: str = "1:1",
    steps: int = 8,
    seed: int = 0,
    num_images: int = 1,
    safety_on: bool = True,
    step_callback: Callable[[int, int], None] | None = None,
    job_id: str | None = None,
) -> GenerationResult:
    job_id = job_id or uuid.uuid4().hex[:12]
    if not prompt or not prompt.strip():
        raise ValueError("Enter a prompt.")
    if aspect_ratio not in ASPECT_DIMS:
        raise ValueError(f"Unknown aspect ratio: {aspect_ratio}")

    lora_path = normalize_lora_path(lora_path)
    width, height = ASPECT_DIMS[aspect_ratio]
    run_started = time.perf_counter()
    log.info(
        "job %s queued model=%s lora=%s ratio=%s size=%sx%s steps=%s seed=%s images=%s safety=%s",
        job_id,
        model,
        lora_path or "none",
        aspect_ratio,
        width,
        height,
        steps,
        seed,
        num_images,
        safety_on,
    )
    with _PIPE_LOCK:
        log.info("job %s started", job_id)
        pipe = get_pipeline(model, lora_path)
        pipe.set_lora_scale(float(lora_strength))
        prepared_at = time.perf_counter()

        def on_step(step: int, total: int):
            log.info("job %s step %s/%s", job_id, step, total)
            if step_callback:
                step_callback(step, total)

        imgs = pipe.generate(
            prompt.strip(),
            width=width,
            height=height,
            steps=int(steps),
            seed=int(seed),
            num_images=int(num_images),
            step_callback=on_step,
        )
        generated_at = time.perf_counter()
        log.info("job %s generated %s image(s) in %.3fs", job_id, len(imgs), generated_at - prepared_at)
        if safety_on:
            from . import safety

            imgs, _ = safety.apply(imgs, enabled=True)
        safety_at = time.perf_counter()
        log.info("job %s safety done in %.3fs", job_id, safety_at - generated_at)
        timings = {
            "prepare_seconds": round(prepared_at - run_started, 3),
            "generate_seconds": round(generated_at - prepared_at, 3),
            "safety_seconds": round(safety_at - generated_at, 3),
            "total_before_save_seconds": round(safety_at - run_started, 3),
            "seconds_per_image": round((generated_at - prepared_at) / max(1, len(imgs)), 3),
            "save_started_at": safety_at,
            "run_started_at": run_started,
        }
        saved, meta_path = save_outputs(
            imgs,
            prompt=prompt.strip(),
            model=model,
            lora_path=lora_path,
            lora_strength=lora_strength,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            safety_on=safety_on,
            timings=timings,
        )
        log.info("job %s saved %s image(s), metadata=%s", job_id, len(saved), meta_path)
    log.info("job %s done total=%.3fs", job_id, timings.get("total_seconds", time.perf_counter() - run_started))
    return GenerationResult(
        job_id=job_id,
        images=imgs,
        saved=saved,
        metadata_path=meta_path,
        timings=timings,
        width=width,
        height=height,
    )
