#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 img2img inference — CLI tool and importable module.

Strategy: GGUF Q5_K_M transformer + torchao INT8 text encoder.
  • No bitsandbytes — torchao is pure PyTorch with native MPS support.
  • GGUF transformer keeps weights in quantized format (~15 GB RAM).
  • torchao INT8 compresses text encoder from ~16 GB → ~8 GB RAM.
    (INT4 is excluded: it requires `mslk >= 1.0.0`, unavailable on macOS.)
  • Total runtime: ~25 GB — fits in 32 GB MacBook Pro Max.

CLI usage:
    python infer.py input.jpg "make the sky a dramatic sunset"
    python infer.py input.jpg "oil painting style" -o result.png
    python infer.py input.jpg "change hair to red" --steps 30 --seed 42
    python infer.py input.jpg "remove the background" --negative "noise, artifacts"

Module usage (imported by app.py):
    from infer import load_pipeline, run_edit
    load_pipeline()
    img = run_edit(pil_image, "make it anime style")
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

DATA_DIR      = Path("./data")
GGUF_PATH     = DATA_DIR / "qwen-image-edit-2511-Q5_K_M.gguf"
PIPELINE_DIR  = DATA_DIR / "Qwen-Image-Edit-2511"
DEFAULT_STEPS = 30
DEFAULT_CFG   = 4.0
DEFAULT_OUT   = Path("./output")

_pipe = None  # module-level singleton


# ── Device ────────────────────────────────────────────────────────────────────

def _best_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Pipeline loading ──────────────────────────────────────────────────────────

def load_pipeline(force_reload: bool = False) -> None:
    """Load model into module-level singleton (idempotent)."""
    global _pipe
    if _pipe is not None and not force_reload:
        return

    from diffusers import (
        GGUFQuantizationConfig,
        QwenImageEditPlusPipeline,
        QwenImageTransformer2DModel,
    )
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    for path, label in [(GGUF_PATH, "GGUF file"), (PIPELINE_DIR, "pipeline dir")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}\nRun ./setup.sh first."
            )

    dev   = _best_device()
    dtype = torch.float16

    # ── Step 1: load pipeline skeleton (text encoder + VAE + scheduler) ───────
    # Pass transformer=None so diffusers skips the missing safetensors shards;
    # we inject the GGUF transformer below once the text encoder is compressed.
    # Using the pipeline's own from_pretrained ensures the text encoder is loaded
    # with the correct class from model_index.json (avoids key-name mismatch).
    print("[infer] Loading pipeline components (text encoder + VAE) ...", file=sys.stderr)
    t0 = time.time()
    _pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(PIPELINE_DIR),
        transformer=None,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    print(f"[infer]   base pipeline ready ({time.time()-t0:.1f}s)", file=sys.stderr)

    # ── Step 2: INT8 quantize text encoder in-place (16 GB → ~8 GB) ──────────
    # INT4 excluded: requires `mslk >= 1.0.0` which is unavailable on macOS.
    # INT8 weight-only works universally on CPU/MPS with no extra dependencies.
    print("[infer] Applying torchao INT8 to text encoder (16 GB → ~8 GB) ...", file=sys.stderr)
    t1 = time.time()
    quantize_(_pipe.text_encoder, Int8WeightOnlyConfig())
    print(f"[infer]   text encoder quantized ({time.time()-t1:.1f}s)", file=sys.stderr)

    # ── Step 3: load GGUF transformer and inject ──────────────────────────────
    print(f"[infer] Loading GGUF transformer ({GGUF_PATH.name}) ...", file=sys.stderr)
    t2 = time.time()
    transformer = QwenImageTransformer2DModel.from_single_file(
        str(GGUF_PATH),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config=str(PIPELINE_DIR),
        subfolder="transformer",
    )
    _pipe.transformer = transformer
    print(f"[infer]   transformer ready ({time.time()-t2:.1f}s)", file=sys.stderr)

    # ── Step 4: memory optimizations ──────────────────────────────────────────
    # AutoencoderKLQwenImage exposes slicing/tiling on the vae directly,
    # not forwarded as pipeline-level helpers.
    _pipe.vae.enable_slicing()
    _pipe.vae.enable_tiling()
    _pipe.enable_attention_slicing()
    # Moves each submodule to MPS only during its forward pass; keeps peak
    # active memory low on Apple unified memory.
    _pipe.enable_model_cpu_offload(device=dev)

    elapsed = time.time() - t0
    print(f"[infer] Pipeline ready on {dev} (total load: {elapsed:.1f}s).", file=sys.stderr)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_edit(
    image: Image.Image,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    steps: int = DEFAULT_STEPS,
    cfg_scale: float = DEFAULT_CFG,
    seed: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> Image.Image:
    """
    Edit a PIL image and return the edited PIL image.

    Args:
        image:           Input PIL image (RGB).
        prompt:          Editing instruction.
        negative_prompt: What to avoid in the output.
        steps:           Diffusion steps (default 30).
        cfg_scale:       Classifier-free guidance (default 4.0).
        seed:            RNG seed for reproducibility.
        height/width:    Output resolution. None = same as input.
    """
    load_pipeline()

    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None

    result = _pipe(
        image=image.convert("RGB"),
        prompt=prompt,
        negative_prompt=negative_prompt or " ",
        num_inference_steps=steps,
        true_cfg_scale=cfg_scale,
        height=height,
        width=width,
        generator=generator,
        output_type="pil",
    )
    return result.images[0]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="infer.py",
        description="Qwen-Image-Edit-2511 img2img on Apple Silicon MPS (no CUDA needed).",
    )
    p.add_argument("input",  help="Input image path")
    p.add_argument("prompt", help="Editing instruction")
    p.add_argument("-o", "--output", default=None, metavar="FILE",
                   help="Output file (default: ./output/edited-<timestamp>.png)")
    p.add_argument("--negative", default=None, metavar="TEXT",
                   help="Negative prompt")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help=f"Diffusion steps (default: {DEFAULT_STEPS})")
    p.add_argument("--cfg", type=float, default=DEFAULT_CFG,
                   help=f"CFG guidance scale (default: {DEFAULT_CFG})")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width",  type=int, default=None)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input image not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_OUT / f"edited-{time.strftime('%Y%m%d-%H%M%S')}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        output_image = run_edit(
            image=Image.open(in_path),
            prompt=args.prompt,
            negative_prompt=args.negative,
            steps=args.steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            height=args.height,
            width=args.width,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    output_image.save(out_path)
    print(f"Done ({time.time() - t0:.1f}s) → {out_path}")
