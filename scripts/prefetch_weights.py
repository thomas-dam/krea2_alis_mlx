#!/usr/bin/env python3
"""Concurrent prefetcher for Krea2 Alis MLX transformer weights.

The app's runtime downloader is intentionally conservative and single-streamed.
This script is for first setup: it uses HTTP Range requests against Hugging Face's
resolve endpoint and writes the final .safetensors file into the repo root, where
app.py/generate.py look before falling back to the cache downloader.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import threading
import time
from pathlib import Path

import requests


BUILDS = {
    "8bit": ("avlp12/Krea-2-Turbo-Alis-MLX-8bit", "transformer_8bit.safetensors"),
    "mixed-4-8": ("avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8", "transformer_mixed_4_8.safetensors"),
}


def _head(url: str) -> tuple[int, bool]:
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    size = int(r.headers["content-length"])
    ranges = r.headers.get("accept-ranges", "").lower() == "bytes"
    return size, ranges


def _load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as f:
        return set(json.load(f))


def _save_done(path: Path, done: set[int]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(sorted(done), f)
    tmp.replace(path)


def download(repo: str, filename: str, out_dir: Path, workers: int, chunk_mb: int) -> Path:
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest = out_dir / filename
    part = out_dir / f"{filename}.part"
    state = out_dir / f"{filename}.ranges.json"

    if dest.exists():
        print(f"already present: {dest}")
        return dest

    size, supports_ranges = _head(url)
    if not supports_ranges:
        raise SystemExit("Server did not advertise Range support; use the app downloader instead.")

    out_dir.mkdir(parents=True, exist_ok=True)
    with part.open("ab") as f:
        f.truncate(size)

    chunk = chunk_mb * 1024 * 1024
    ranges = [(start, min(start + chunk, size) - 1) for start in range(0, size, chunk)]
    done = _load_done(state)
    lock = threading.Lock()
    downloaded = sum((end - start + 1) for i, (start, end) in enumerate(ranges) if i in done)
    started_at = time.monotonic()
    last_shown = 0.0

    def show() -> None:
        nonlocal last_shown
        now = time.monotonic()
        if now - last_shown < 1 and downloaded < size:
            return
        last_shown = now
        elapsed = max(time.monotonic() - started_at, 0.001)
        mb_done = downloaded / 1048576
        mb_total = size / 1048576
        rate = mb_done / elapsed
        print(f"\r  ↓ {filename}  {mb_done:,.0f} / {mb_total:,.0f} MB  {rate:,.1f} MB/s", end="", flush=True)

    def fetch(i: int, start: int, end: int) -> None:
        nonlocal downloaded
        if i in done:
            return
        headers = {"Range": f"bytes={start}-{end}"}
        with requests.get(url, headers=headers, stream=True, timeout=(30, 180), allow_redirects=True) as r:
            if r.status_code != 206:
                raise RuntimeError(f"range {start}-{end} returned HTTP {r.status_code}")
            pos = start
            with part.open("r+b") as f:
                f.seek(pos)
                for block in r.iter_content(1024 * 1024):
                    if not block:
                        continue
                    f.write(block)
                    pos += len(block)
                    with lock:
                        downloaded += len(block)
                        show()
            if pos != end + 1:
                raise RuntimeError(f"range {start}-{end} ended at {pos}")
        with lock:
            done.add(i)
            _save_done(state, done)

    pending = [(i, start, end) for i, (start, end) in enumerate(ranges) if i not in done]
    print(f"downloading {repo}/{filename} with {workers} workers")
    show()
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch, i, start, end) for i, start, end in pending]
        for job in futures.as_completed(jobs):
            job.result()
    print()

    if len(done) != len(ranges):
        raise RuntimeError("download incomplete; rerun to resume")
    if part.stat().st_size != size:
        raise RuntimeError(f"download size mismatch: {part.stat().st_size} != {size}")
    part.replace(dest)
    state.unlink(missing_ok=True)
    print(f"saved {dest}")
    return dest


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=BUILDS, default="mixed-4-8")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=128)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args(argv)

    repo, filename = BUILDS[args.precision]
    download(repo, filename, Path(args.out_dir), args.workers, args.chunk_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
