#!/usr/bin/env python3
"""
FastAPI web app for Qwen-Image-Edit-2511 img2img inference.
Delegates all model work to infer.py (same logic as the CLI).

Run:
    python app.py
    python app.py --port 9000 --steps 30
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen-Image-Edit FastAPI web app")
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--port",  type=int, default=8000)
    p.add_argument("--steps", type=int, default=20,
                   help="Default diffusion steps (overridable per request)")
    return p.parse_args()


_args = _parse_args()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
MAX_MB = 20

app = FastAPI(title="Qwen-Image-Edit Demo", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    from infer import load_pipeline
    load_pipeline()


# ── HTML UI ───────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Qwen-Image-Edit · img2img</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --accent:  #58a6ff;
    --accent2: #bc8cff;
    --text:    #e6edf3;
    --muted:   #8b949e;
    --r:       10px;
  }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 2rem 1rem 5rem;
  }
  header { text-align: center; margin-bottom: 2rem; }
  header h1 {
    font-size: 1.85rem; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  header p { color: var(--muted); margin-top: .3rem; font-size: .88rem; }
  .card {
    width: 100%; max-width: 920px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r); padding: 1.5rem;
  }

  /* ── Before / After ── */
  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem; }
  @media (max-width: 620px) { .panels { grid-template-columns: 1fr; } }
  .panel-label { font-size: .75rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: .07em; margin-bottom: .45rem; }

  /* Drop zone */
  #drop-zone {
    border: 2px dashed var(--border); border-radius: var(--r);
    min-height: 240px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    cursor: pointer; transition: border-color .2s, background .2s;
    position: relative; overflow: hidden; gap: .5rem; padding: 1rem;
    text-align: center;
  }
  #drop-zone.drag-over { border-color: var(--accent); background: rgba(88,166,255,.06); }
  #drop-zone input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  #drop-zone .icon { font-size: 2rem; }
  #drop-zone p { color: var(--muted); font-size: .86rem; }

  #preview-wrap { display: none; position: relative; }
  .img-box {
    width: 100%; height: 240px; object-fit: contain;
    border-radius: 8px; border: 1px solid var(--border); background: #0a0d12;
    display: block;
  }
  #clear-btn {
    position: absolute; top: 6px; right: 6px;
    background: rgba(0,0,0,.65); border: none; border-radius: 50%;
    color: #fff; width: 26px; height: 26px;
    cursor: pointer; font-size: .85rem;
    display: flex; align-items: center; justify-content: center;
  }
  #clear-btn:hover { background: rgba(200,50,50,.7); }

  /* Output panel */
  #output-panel {
    border: 1px solid var(--border); border-radius: var(--r);
    min-height: 240px; background: #0a0d12;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    overflow: hidden; gap: .6rem; padding: .5rem;
  }
  #output-placeholder { color: var(--muted); font-size: .86rem; text-align: center; padding: 1rem; }
  #output-img { width: 100%; height: 240px; object-fit: contain; border-radius: 6px; display: none; }
  #dl-btn {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    color: var(--accent); font-size: .8rem; padding: .28rem .7rem;
    cursor: pointer; text-decoration: none; display: none;
  }
  #dl-btn:hover { border-color: var(--accent); }

  /* Prompt */
  .field-label { display: block; font-size: .75rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: .07em; margin: 1rem 0 .4rem; }
  textarea, input[type="text"], input[type="number"], select {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); font-size: .92rem;
    padding: .6rem .85rem; font-family: inherit; transition: border-color .2s;
  }
  textarea { resize: vertical; min-height: 68px; }
  textarea:focus, input:focus, select:focus { outline: none; border-color: var(--accent); }

  /* Chips */
  .chips { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .5rem; }
  .chip {
    background: var(--bg); border: 1px solid var(--border); border-radius: 20px;
    padding: .22rem .62rem; font-size: .76rem; color: var(--muted);
    cursor: pointer; user-select: none; transition: border-color .15s, color .15s;
  }
  .chip:hover { border-color: var(--accent); color: var(--text); }

  /* Advanced */
  details { margin-top: 1rem; }
  summary { font-size: .82rem; color: var(--muted); cursor: pointer; user-select: none;
    list-style: none; display: flex; align-items: center; gap: .4rem; }
  summary::-webkit-details-marker { display: none; }
  summary::before { content: "▶"; font-size: .58rem; transition: transform .18s; }
  details[open] summary::before { transform: rotate(90deg); }
  .opts { display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
    gap: .7rem; margin-top: .8rem; }
  .opt label { display: block; font-size: .75rem; color: var(--muted); margin-bottom: .28rem; }
  .opt input, .opt select { padding: .42rem .65rem; font-size: .86rem; border-radius: 6px; }
  .opt-wide { grid-column: span 2; }
  @media (max-width: 500px) { .opt-wide { grid-column: span 1; } }

  /* Submit */
  #submit-btn {
    width: 100%; margin-top: 1.2rem; padding: .78rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none; border-radius: 8px; color: #fff;
    font-size: .96rem; font-weight: 600; cursor: pointer;
    transition: opacity .2s, transform .1s;
  }
  #submit-btn:disabled { opacity: .38; cursor: not-allowed; transform: none; }
  #submit-btn:not(:disabled):hover { opacity: .88; }
  #submit-btn:not(:disabled):active { transform: scale(.98); }

  /* Spinner */
  #spinner { display: none; flex-direction: column; align-items: center;
    gap: .7rem; padding: 1.25rem 0 .25rem; }
  .ring {
    width: 34px; height: 34px; border: 3px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #spinner p { color: var(--muted); font-size: .86rem; text-align: center; }

  /* Error */
  #err { display: none; margin-top: .85rem; background: rgba(248,81,73,.1);
    border: 1px solid rgba(248,81,73,.3); border-radius: 8px;
    padding: .6rem .85rem; color: #f85149; font-size: .86rem; }

  #timing { font-size: .74rem; color: var(--muted); text-align: right; margin-top: .4rem; }
</style>
</head>
<body>

<header>
  <h1>Qwen-Image-Edit · img2img</h1>
  <p>Q5_K_M GGUF · Apple Silicon MPS · Real pixel-level image editing</p>
</header>

<div class="card">

  <!-- Before / After panels -->
  <div class="panels">
    <div>
      <div class="panel-label">Input image</div>
      <div id="drop-zone">
        <input type="file" id="file-input" accept="image/*"/>
        <div class="icon">🖼️</div>
        <p>Drop image here or <strong>click to browse</strong><br/>
          <span style="font-size:.76rem">JPEG · PNG · WEBP — max 20 MB</span></p>
      </div>
      <div id="preview-wrap">
        <img id="preview-img" class="img-box" alt="Input"/>
        <button id="clear-btn" title="Remove">✕</button>
      </div>
    </div>
    <div>
      <div class="panel-label">Output image</div>
      <div id="output-panel">
        <div id="output-placeholder">Output will appear here</div>
        <img id="output-img" class="img-box" alt="Edited output"/>
        <a id="dl-btn" download="edited.png">⬇ Download</a>
      </div>
    </div>
  </div>

  <!-- Prompt -->
  <label class="field-label" for="prompt-input">Editing instruction</label>
  <textarea id="prompt-input" rows="2"
    placeholder="e.g. "make the sky a dramatic sunset", "oil painting style", "change hair to red""></textarea>

  <div class="chips">
    <span class="chip" data-p="make the sky a dramatic sunset with orange and purple tones">Dramatic sunset</span>
    <span class="chip" data-p="change the hair color to vibrant red">Red hair</span>
    <span class="chip" data-p="convert to oil painting style with visible brushstrokes">Oil painting</span>
    <span class="chip" data-p="add snow to the scene, make it look like winter">Add snow</span>
    <span class="chip" data-p="remove the background, replace with clean white">Remove BG</span>
    <span class="chip" data-p="apply golden hour warm lighting tones">Golden hour</span>
    <span class="chip" data-p="convert to anime illustration style">Anime style</span>
    <span class="chip" data-p="apply cinematic teal and orange color grading">Cinematic grade</span>
  </div>

  <!-- Advanced options -->
  <details>
    <summary>Advanced options</summary>
    <div class="opts">
      <div class="opt">
        <label>Steps (default 20)</label>
        <input type="number" id="opt-steps" value="20" min="1" max="100"/>
      </div>
      <div class="opt">
        <label>CFG scale (default 4.0)</label>
        <input type="number" id="opt-cfg" value="4.0" step="0.5" min="1" max="15"/>
      </div>
      <div class="opt">
        <label>Seed (blank = random)</label>
        <input type="number" id="opt-seed" placeholder="e.g. 42"/>
      </div>
      <div class="opt">
        <label>Output size</label>
        <select id="opt-size">
          <option value="">Same as input</option>
          <option value="512">512 × 512</option>
          <option value="768">768 × 768</option>
          <option value="1024">1024 × 1024</option>
        </select>
      </div>
      <div class="opt opt-wide">
        <label>Negative prompt</label>
        <input type="text" id="opt-neg" placeholder="blurry, watermark, artifacts, low quality"/>
      </div>
    </div>
  </details>

  <button id="submit-btn" disabled>Edit Image</button>

  <div id="spinner">
    <div class="ring"></div>
    <p>Running inference on MPS — may take several minutes on 32 GB RAM…</p>
  </div>

  <div id="err"></div>
  <div id="timing"></div>

</div>

<script>
const fileInput = document.getElementById('file-input');
const dropZone  = document.getElementById('drop-zone');
const prevWrap  = document.getElementById('preview-wrap');
const prevImg   = document.getElementById('preview-img');
const clearBtn  = document.getElementById('clear-btn');
const promptEl  = document.getElementById('prompt-input');
const submitBtn = document.getElementById('submit-btn');
const spinner   = document.getElementById('spinner');
const outImg    = document.getElementById('output-img');
const outHolder = document.getElementById('output-placeholder');
const dlBtn     = document.getElementById('dl-btn');
const errEl     = document.getElementById('err');
const timingEl  = document.getElementById('timing');

let selectedFile = null;

function setFile(f) {
  if (!f || !f.type.startsWith('image/')) return;
  if (f.size > 20 * 1024 * 1024) { showErr('File too large — max 20 MB.'); return; }
  selectedFile = f;
  prevImg.src = URL.createObjectURL(f);
  prevWrap.style.display = 'block';
  dropZone.style.display = 'none';
  sync();
}
function clearFile() {
  selectedFile = null;
  prevWrap.style.display = 'none';
  dropZone.style.display = 'flex';
  fileInput.value = '';
  sync();
}
function sync() { submitBtn.disabled = !(selectedFile && promptEl.value.trim()); }
function showErr(msg) { errEl.textContent = msg; errEl.style.display = 'block'; }
function clearErr() { errEl.style.display = 'none'; }

fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
clearBtn.addEventListener('click', clearFile);
promptEl.addEventListener('input', sync);

document.querySelector('.chips').addEventListener('click', e => {
  const c = e.target.closest('.chip');
  if (c) { promptEl.value = c.dataset.p; sync(); promptEl.focus(); }
});

submitBtn.addEventListener('click', async () => {
  clearErr();
  timingEl.textContent = '';
  outImg.style.display = 'none';
  outHolder.style.display = 'flex';
  dlBtn.style.display = 'none';
  spinner.style.display = 'flex';
  submitBtn.disabled = true;

  const fd = new FormData();
  fd.append('image', selectedFile);
  fd.append('prompt', promptEl.value.trim());
  fd.append('steps',  document.getElementById('opt-steps').value || 20);
  fd.append('cfg_scale', document.getElementById('opt-cfg').value || 4.0);
  const seed = document.getElementById('opt-seed').value;
  if (seed) fd.append('seed', seed);
  const neg = document.getElementById('opt-neg').value;
  if (neg)  fd.append('negative_prompt', neg);
  const size = document.getElementById('opt-size').value;
  if (size) { fd.append('height', size); fd.append('width', size); }

  const t0 = performance.now();
  try {
    const res  = await fetch('/inference', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    const src = `data:image/png;base64,${data.image_b64}`;
    outImg.src = src;
    outImg.style.display = 'block';
    outHolder.style.display = 'none';
    dlBtn.href = src;
    dlBtn.style.display = 'inline-block';
    timingEl.textContent = `Completed in ${((performance.now()-t0)/1000).toFixed(1)}s`;
  } catch (e) {
    showErr(`Inference failed: ${e.message}`);
  } finally {
    spinner.style.display = 'none';
    sync();
  }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


# ── Inference endpoint ────────────────────────────────────────────────────────

@app.post("/inference")
async def inference(
    image:           UploadFile = File(...),
    prompt:          str        = Form(...),
    steps:           int        = Form(default=50),
    cfg_scale:       float      = Form(default=4.0),
    seed:            int        = Form(default=None),
    negative_prompt: str        = Form(default=None),
    height:          int        = Form(default=None),
    width:           int        = Form(default=None),
) -> JSONResponse:
    from infer import run_edit

    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{image.content_type}'. "
                   f"Accepted: {', '.join(ALLOWED_TYPES)}",
        )
    data = await image.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_MB} MB.")

    input_image = Image.open(io.BytesIO(data)).convert("RGB")

    output_image = run_edit(
        image=input_image,
        prompt=prompt.strip(),
        negative_prompt=negative_prompt or None,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        height=height,
        width=width,
    )

    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({"image_b64": img_b64})


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    import torch
    from pathlib import Path
    return {
        "status":  "ok",
        "model":   str(Path("./data/qwen-image-edit-4bit").resolve()),
        "device":  "mps" if torch.backends.mps.is_available() else "cpu",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")
