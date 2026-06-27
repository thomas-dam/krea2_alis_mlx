"""Content-safety filter (NSFW image classifier).

The Krea 2 Community License (§4.2) requires deployments to implement reasonable content
filtering. This uses **Falconsai/nsfw_image_detection** — one of the classifiers the license
itself lists as an example — to flag explicit outputs, which are then redacted.

On by default; disable per-call (`enabled=False`, the CLI `--no-safety` flag, or the web-UI
toggle) or globally with `KREA2_DISABLE_SAFETY=1`. Threshold via `KREA2_SAFETY_THRESHOLD`
(default 0.85, tuned so ordinary swimwear/beach photos pass). Degrades gracefully — if the
classifier can't be loaded it warns and lets generation continue unfiltered.
"""

from __future__ import annotations

import os

NSFW_REPO = "Falconsai/nsfw_image_detection"
_THRESHOLD = float(os.environ.get("KREA2_SAFETY_THRESHOLD", "0.85"))
_PIPE = None
_FAILED = False


def _ensure_model() -> str:
    """Download the classifier's files via our HTTP bridge (it's Xet-backed, so the default
    huggingface_hub path can hang behind firewalls — same fix as the main model)."""
    from huggingface_hub import HfApi

    from .pipeline import _CACHE, _http_download

    dest = os.path.join(_CACHE, NSFW_REPO.replace("/", "__"))
    want = [
        s.rfilename for s in HfApi().model_info(NSFW_REPO).siblings
        if s.rfilename.endswith((".safetensors", ".json", ".txt"))
    ]
    if not any(f.endswith(".safetensors") for f in want):  # fall back to PyTorch weights if no safetensors
        want += [s.rfilename for s in HfApi().model_info(NSFW_REPO).siblings if s.rfilename.endswith(".bin")]
    for f in want:
        _http_download(NSFW_REPO, f, dest)
    return dest


def _classifier():
    global _PIPE, _FAILED
    if _PIPE is None and not _FAILED:
        try:
            from transformers import pipeline
            _PIPE = pipeline("image-classification", model=_ensure_model())
        except Exception as e:  # network, missing dep, etc. — never block generation
            _FAILED = True
            print(f"[safety] NSFW classifier unavailable ({type(e).__name__}: {e}); "
                  f"continuing WITHOUT filtering. Use your own filter for public deployments.")
    return _PIPE


def is_nsfw(image, threshold: float = _THRESHOLD) -> bool:
    clf = _classifier()
    if clf is None:
        return False
    try:
        scores = {d["label"].lower(): d["score"] for d in clf(image)}
        return scores.get("nsfw", 0.0) >= threshold
    except Exception:
        return False


def _redact(image):
    """Replace a flagged image with a flat placeholder (decisive — no discernible content)."""
    from PIL import Image, ImageDraw

    out = Image.new("RGB", image.size, (28, 28, 32))
    d = ImageDraw.Draw(out)
    msg = "⚠  flagged by safety filter"
    try:
        w = d.textlength(msg)
    except Exception:
        w = 8 * len(msg)
    d.text(((image.size[0] - w) / 2, image.size[1] / 2 - 8), msg, fill=(200, 200, 210))
    return out


def apply(images, enabled: bool = True, threshold: float = _THRESHOLD):
    """Return (images, n_flagged); flagged images are redacted."""
    if not enabled or os.environ.get("KREA2_DISABLE_SAFETY"):
        return list(images), 0
    out, flagged = [], 0
    for im in images:
        if is_nsfw(im, threshold):
            out.append(_redact(im))
            flagged += 1
        else:
            out.append(im)
    return out, flagged
