#!/usr/bin/env python
"""FastAPI entrypoint for headless Krea2 generation.

Run:
    .venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 7861
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from krea2.service import ASPECT_DIMS, ASPECT_RATIOS, MODELS, default_precision, generate_and_save


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="Krea2 Alis MLX API", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "8bit"
    lora_path: str | None = None
    lora_strength: float = 1.0
    aspect_ratio: str = "1:1"
    steps: int = Field(8, ge=1, le=50)
    seed: int = Field(0, ge=0)
    num_images: int = Field(1, ge=1, le=8)
    safety_on: bool = True


class GenerateResponse(BaseModel):
    job_id: str
    images: list[str]
    metadata: str
    timings: dict
    width: int
    height: int


@app.get("/health")
def health():
    return {"ok": True, "default_model": default_precision()}


@app.get("/ratios")
def ratios():
    return [{"label": label, "value": value, "width": ASPECT_DIMS[value][0], "height": ASPECT_DIMS[value][1]} for label, value in ASPECT_RATIOS]


@app.get("/models")
def models():
    return [{"label": label, "value": value} for label, value in MODELS]


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        result = generate_and_save(
            prompt=req.prompt,
            model=req.model,
            lora_path=req.lora_path,
            lora_strength=req.lora_strength,
            aspect_ratio=req.aspect_ratio,
            steps=req.steps,
            seed=req.seed,
            num_images=req.num_images,
            safety_on=req.safety_on,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        m = str(e).lower()
        if any(k in m for k in ("memory", "alloc", "metal")):
            raise HTTPException(
                status_code=507,
                detail="Out of memory. Try fewer images, a smaller aspect preset, or the mixed-4/8 build.",
            ) from None
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from None
    return GenerateResponse(
        job_id=result.job_id,
        images=result.saved,
        metadata=result.metadata_path,
        timings=result.timings,
        width=result.width,
        height=result.height,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7861)
