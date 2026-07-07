#!/usr/bin/env python
"""Local web UI for Krea-2-Turbo on Apple MLX.

    python3 -m pip install -r requirements.txt
    python3 app.py            # opens http://localhost:7860

Auto-detects the weights shipped in this folder (8-bit or mixed-4/8). The VAE /
Qwen3-VL-4B encoder / tokenizer are pulled from krea/Krea-2-Turbo on first run.
Set KREA2_BASE_DIR to a local Krea-2-Turbo snapshot to skip that download.
"""

import os
from pathlib import Path

import gradio as gr

from krea2.service import ASPECT_RATIOS, MODELS, default_precision, generate_and_save

LORAS_DIR = Path(__file__).resolve().parent / "loras"


KEYBOARD_SHORTCUT_JS = """
function() {
  document.addEventListener("keydown", function(event) {
    if (event.repeat || event.key !== "Enter" || (!event.ctrlKey && !event.metaKey)) {
      return;
    }
    const button = document.querySelector("#generate-button button") || document.querySelector("#generate-button");
    if (button && !button.disabled) {
      event.preventDefault();
      button.click();
    }
  });
}
"""


def lora_choices():
    LORAS_DIR.mkdir(exist_ok=True)
    files = sorted(LORAS_DIR.glob("*.safetensors"), key=lambda p: p.name.lower())
    return [("None", "")] + [(p.stem, str(p)) for p in files]


def generate(
    prompt,
    model,
    lora_path,
    lora_strength,
    lora_path_2,
    lora_strength_2,
    reference_image,
    reference_strength,
    depth_image,
    depth_lora_path,
    depth_strength,
    aspect_ratio,
    steps,
    seed,
    num_images,
    safety_on,
    progress=gr.Progress(),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    try:
        progress(0, desc="Preparing selected model…")
        reference_strength = float(reference_strength)
        if reference_image is not None and reference_strength > 0:
            init_image = reference_image
            # The sampler uses img2img "change strength": lower values preserve the input,
            # higher values re-imagine it. The UI exposes the inverse: reference strength.
            init_strength = max(1.0 - reference_strength, 1e-6)
        else:
            init_image = None
            init_strength = 1.0

        def cb(step, total):
            progress(step / total, desc=f"Generating · step {step}/{total}")

        result = generate_and_save(
            prompt=prompt,
            model=model,
            lora_path=lora_path,
            lora_strength=float(lora_strength),
            lora_path_2=lora_path_2,
            lora_strength_2=float(lora_strength_2),
            init_image=init_image,
            init_strength=init_strength,
            depth_image=depth_image,
            depth_lora_path=depth_lora_path,
            depth_strength=float(depth_strength),
            aspect_ratio=aspect_ratio,
            steps=int(steps),
            seed=int(seed),
            num_images=int(num_images),
            safety_on=bool(safety_on),
            step_callback=cb,
        )
        progress(1.0, desc="Saving outputs…")
        timings = result.timings
        timing_text = (
            f"Job: {result.job_id}\n"
            f"Timing: total {timings['total_seconds']}s · "
            f"generate {timings['generate_seconds']}s · "
            f"{timings['seconds_per_image']}s/image"
        )
        return result.images, timing_text + "\n\nSaved:\n" + "\n".join(result.saved + [result.metadata_path])
    except gr.Error:
        raise
    except Exception as e:  # surface OOM / download errors as a friendly message, not a traceback
        m = str(e).lower()
        if any(k in m for k in ("memory", "alloc", "metal")):
            raise gr.Error("Out of memory — these 1024-base presets need ~24 GB+ unified memory. "
                           "Try fewer Images or the mixed-4/8 build.") from None
        raise gr.Error(f"Generation failed: {e}") from None


with gr.Blocks(title="Krea 2 Turbo · Alis MLX", js=KEYBOARD_SHORTCUT_JS) as demo:
    default_prec = default_precision()  # the build already in this folder, if any
    gr.Markdown("# Krea&nbsp;2&nbsp;Turbo · Alis MLX\n"
                "Local text-to-image on Apple silicon · 8-step Turbo (no CFG). "
                "**First run loads the model (~30 s); then ~50 s per 1024-base image on an M3 Ultra** "
                "(slower chips take longer; ×N for N images).")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=3, value="a fox in the snow")
            model = gr.Dropdown(MODELS, value=default_prec, label="Model")
            lora_path = gr.Dropdown(lora_choices(), value="", label="LoRA 1")
            lora_strength = gr.Slider(-10.0, 10.0, value=1.0, step=0.05, label="LoRA 1 strength")
            lora_path_2 = gr.Dropdown(lora_choices(), value="", label="LoRA 2")
            lora_strength_2 = gr.Slider(-10.0, 10.0, value=1.0, step=0.05, label="LoRA 2 strength")
            reference_image = gr.Image(label="Reference image", type="pil")
            reference_strength = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Reference strength")
            depth_image = gr.Image(label="Depth map", type="pil")
            depth_lora_path = gr.Dropdown(lora_choices(), value="", label="Depth control LoRA")
            depth_strength = gr.Slider(0.0, 10.0, value=1.0, step=0.05, label="Depth strength")
            with gr.Row():
                aspect_ratio = gr.Dropdown(ASPECT_RATIOS, value="1:1", label="Aspect ratio")
                steps = gr.Slider(4, 12, value=8, step=1, label="Steps")
            with gr.Row():
                seed = gr.Number(value=0, label="Seed", precision=0)
                num_images = gr.Slider(1, 4, value=1, step=1, label="Images")
            safety_chk = gr.Checkbox(value=True, label="NSFW safety filter (recommended; required by the license for public deployments)")
            btn = gr.Button("Generate", variant="primary", elem_id="generate-button")
            gr.Examples(
                [["a fox in the snow"],
                 ["a neon city street at night in the rain, reflections"],
                 ["a close-up portrait of an old fisherman, weathered face"]],
                inputs=prompt,
            )
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Output", columns=2, height=560, object_fit="contain")
            saved_paths = gr.Textbox(label="Saved files", lines=5, interactive=False)
    btn.click(
        generate,
        [
            prompt,
            model,
            lora_path,
            lora_strength,
            lora_path_2,
            lora_strength_2,
            reference_image,
            reference_strength,
            depth_image,
            depth_lora_path,
            depth_strength,
            aspect_ratio,
            steps,
            seed,
            num_images,
            safety_chk,
        ],
        [gallery, saved_paths],
    )


if __name__ == "__main__":
    # bind to loopback by default (don't expose the generator on the LAN); override with KREA2_HOST
    demo.queue().launch(server_name=os.environ.get("KREA2_HOST", "127.0.0.1"), theme=gr.themes.Soft())
