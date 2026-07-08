#!/usr/bin/env python
"""Local web UI for Krea-2-Turbo on Apple MLX.

    python3 -m pip install -r requirements.txt
    python3 app.py            # opens http://localhost:7860

Auto-detects the weights shipped in this folder (8-bit or mixed-4/8). The VAE /
Qwen3-VL-4B encoder / tokenizer are pulled from krea/Krea-2-Turbo on first run.
Set KREA2_BASE_DIR to a local Krea-2-Turbo snapshot to skip that download.
"""

import os
import random
from pathlib import Path

import gradio as gr

from krea2.service import ASPECT_RATIOS, MODELS, default_precision, generate_and_save

LORAS_DIR = Path(__file__).resolve().parent / "loras"


# Gradio 6 injects this string verbatim as a <script> tag (it no longer wraps or
# invokes it), so it must be a self-executing statement, not a bare function.
KEYBOARD_SHORTCUT_JS = """
(() => {
  document.addEventListener("keydown", (event) => {
    if (event.repeat || event.key !== "Enter" || (!event.ctrlKey && !event.metaKey)) {
      return;
    }
    const button = document.querySelector("#generate-button button") || document.querySelector("#generate-button");
    if (button && !button.disabled) {
      event.preventDefault();
      button.click();
    }
  });
})();
"""

# Tab layout with a fixed generate bar; content gets bottom padding so the bar
# never covers it.
CSS = """
.gradio-container {max-width: 920px !important; margin: 0 auto; padding-bottom: 120px !important;}
#genbar {
  position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
  background: var(--background-fill-primary);
  border-top: 1px solid var(--border-color-primary);
  box-shadow: 0 -4px 20px rgba(0,0,0,.08);
  padding: 10px max(24px, calc((100vw - 872px) / 2));
  margin: 0; align-items: center; gap: 12px;
}
#genbar > * {min-width: 0;}
#generate-button {height: 46px; font-size: 1.05em; font-weight: 700;}
#genbar-status, #genbar-status label, #genbar-status input, #genbar-status textarea {
  border: none !important; background: transparent !important; box-shadow: none !important;
}
#genbar-status input, #genbar-status textarea {
  font-family: var(--font-mono); font-size: 11px; color: var(--body-text-color-subdued);
}
/* compact progress readout while generating: one line of text, thin bar, small timer */
#genbar-status .wrap {background: transparent; padding: 0; inset: 0;}
#genbar-status .eta-bar {display: none;}
#genbar-status .progress-level {width: 100%;}
#genbar-status .progress-level-inner {
  margin: 0; max-width: 100%; font-size: 11px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#genbar-status .progress-bar-wrap {width: 100%; height: 5px; border-radius: 999px; overflow: hidden;}
#genbar-status .progress-text {
  position: absolute; right: 2px; top: 50%; transform: translateY(-50%);
  font-size: 10px; font-family: var(--font-mono); z-index: 3;
  background: var(--background-fill-primary); padding: 0 4px; border-radius: 4px;
}
/* hide the stale status text while the live progress overlay is on top of it */
#genbar-status:has(.wrap:not(.hide)) textarea {opacity: 0;}
#seed-random {align-self: stretch; height: auto;}
#app-header {display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; padding: 4px 0 0;}
#app-header h1 {font-size: 20px; margin: 0;}
#app-header .sub {color: var(--body-text-color-subdued); font-size: 12px;}
"""


def lora_choices():
    LORAS_DIR.mkdir(exist_ok=True)
    files = sorted(LORAS_DIR.glob("*.safetensors"), key=lambda p: p.name.lower())
    return [("None", "")] + [(p.stem, str(p)) for p in files]


def random_seed():
    return random.randint(0, 2**31 - 1)


def lora_badge(slot, path, strength):
    if path:
        return gr.Accordion(label=f"LoRA {slot} · {Path(path).stem} · {float(strength):g}")
    return gr.Accordion(label=f"LoRA {slot} · off")


def reference_badge(image):
    return gr.Accordion(label="🖼 Reference image · on" if image is not None else "🖼 Reference image · off")


def depth_badge(image):
    return gr.Accordion(label="🗺 Depth map · on" if image is not None else "🗺 Depth map · off")


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
    session_images,
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
        # Newest images first; the session gallery accumulates across jobs.
        session_images = list(result.images) + list(session_images or [])
        return (
            session_images,
            timing_text + "\n\nSaved:\n" + "\n".join(result.saved + [result.metadata_path]),
            f"done · {timings['total_seconds']}s · {len(session_images)} image(s) this session",
            session_images,
        )
    except gr.Error:
        raise
    except Exception as e:  # surface OOM / download errors as a friendly message, not a traceback
        m = str(e).lower()
        if any(k in m for k in ("memory", "alloc", "metal")):
            raise gr.Error("Out of memory — these 1024-base presets need ~24 GB+ unified memory. "
                           "Try fewer Images or the mixed-4/8 build.") from None
        raise gr.Error(f"Generation failed: {e}") from None


with gr.Blocks(title="Krea 2 Turbo · Alis MLX") as demo:
    default_prec = default_precision()  # the build already in this folder, if any
    session_images = gr.State([])

    gr.HTML(
        '<div id="app-header"><h1>Krea&nbsp;2&nbsp;Turbo <span style="font-weight:400;opacity:.6">· Alis MLX</span></h1>'
        '<span class="sub">Local text-to-image on Apple silicon · 8-step Turbo (no CFG) · '
        'first run loads the model (~30 s), then ~50 s per 1024-base image on an M3 Ultra</span></div>'
    )

    with gr.Tabs() as tabs:
        # ---------------- Tab 1: Prompt & settings ----------------
        with gr.Tab("✏️ Prompt", id="prompt"):
            prompt = gr.Textbox(label="Prompt", lines=6, value="a fox in the snow",
                                placeholder="Describe the image you want…")
            gr.Examples(
                [["a fox in the snow"],
                 ["a neon city street at night in the rain, reflections"],
                 ["a close-up portrait of an old fisherman, weathered face"]],
                inputs=prompt,
            )
            with gr.Group():
                with gr.Row():
                    model = gr.Dropdown(MODELS, value=default_prec, label="Model")
                    aspect_ratio = gr.Dropdown(ASPECT_RATIOS, value="1:1", label="Aspect ratio")
                    steps = gr.Slider(4, 12, value=8, step=1, label="Steps")
                with gr.Row():
                    seed = gr.Number(value=0, label="Seed", precision=0, scale=2)
                    seed_btn = gr.Button("🎲 Random seed", scale=1, elem_id="seed-random")
                safety_chk = gr.Checkbox(
                    value=True,
                    label="NSFW safety filter (recommended; required by the license for public deployments)",
                )

        # ---------------- Tab 2: LoRAs & image control ----------------
        with gr.Tab("🎛 LoRAs & Control", id="loras"):
            with gr.Accordion("LoRA 1 · off", open=True) as lora_acc_1:
                lora_path = gr.Dropdown(lora_choices(), value="", label="LoRA")
                lora_strength = gr.Slider(-10.0, 10.0, value=1.0, step=0.05, label="Strength")
            with gr.Accordion("LoRA 2 · off", open=False) as lora_acc_2:
                lora_path_2 = gr.Dropdown(lora_choices(), value="", label="LoRA")
                lora_strength_2 = gr.Slider(-10.0, 10.0, value=1.0, step=0.05, label="Strength")
            with gr.Accordion("🖼 Reference image · off", open=False) as ref_acc:
                reference_image = gr.Image(label="Reference image", type="pil")
                reference_strength = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Reference strength")
            with gr.Accordion("🗺 Depth map · off", open=False) as depth_acc:
                depth_image = gr.Image(label="Depth map", type="pil")
                with gr.Row():
                    depth_lora_path = gr.Dropdown(lora_choices(), value="", label="Depth control LoRA")
                    depth_strength = gr.Slider(0.0, 10.0, value=1.0, step=0.05, label="Depth strength")

        # ---------------- Tab 3: Gallery ----------------
        with gr.Tab("🖼 Gallery", id="gallery"):
            gallery = gr.Gallery(label="Session gallery", columns=3, height=560, object_fit="contain")
            with gr.Accordion("Last job details", open=False):
                saved_paths = gr.Textbox(label="Saved files", lines=5, interactive=False)

    # ---------------- Sticky generate bar (visible from every tab) ----------------
    with gr.Row(elem_id="genbar"):
        num_images = gr.Slider(1, 4, value=1, step=1, label="Images", scale=1, container=True)
        status = gr.Textbox(value="ready", show_label=False, interactive=False, lines=1,
                            container=False, elem_id="genbar-status", scale=2)
        btn = gr.Button("⚡ Generate", variant="primary", elem_id="generate-button", scale=1)

    # keep accordion badges in sync with their controls
    for src in (lora_path, lora_strength):
        src.change(lambda p, s: lora_badge(1, p, s), [lora_path, lora_strength], lora_acc_1,
                   show_progress="hidden")
    for src in (lora_path_2, lora_strength_2):
        src.change(lambda p, s: lora_badge(2, p, s), [lora_path_2, lora_strength_2], lora_acc_2,
                   show_progress="hidden")
    reference_image.change(reference_badge, reference_image, ref_acc, show_progress="hidden")
    depth_image.change(depth_badge, depth_image, depth_acc, show_progress="hidden")

    seed_btn.click(random_seed, None, seed, show_progress="hidden")

    # In Gradio 6.19, returning a gr.Tabs(selected=...) update from the same event
    # that updates progress-tracked components leaves the progress overlay stuck at
    # 100%, so the jump to the Gallery tab runs as a chained follow-up event instead.
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
            session_images,
        ],
        [gallery, saved_paths, status, session_images],
    ).then(lambda: gr.Tabs(selected="gallery"), None, [tabs], show_progress="hidden")


if __name__ == "__main__":
    # bind to loopback by default (don't expose the generator on the LAN); override with KREA2_HOST
    demo.queue().launch(
        server_name=os.environ.get("KREA2_HOST", "127.0.0.1"),
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
        js=KEYBOARD_SHORTCUT_JS,
    )
