#!/usr/bin/env bash
# Setup: Python 3.11 venv → install deps → download models into ./data/
#
# Strategy: GGUF Q5_K_M transformer + torchao INT4 text encoder
#   No bitsandbytes needed — torchao is pure PyTorch and works on MPS.
#
# Download breakdown (stored in ./data/):
#   GGUF transformer   Q5_K_M           ~15.0 GB   unsloth/Qwen-Image-Edit-2511-GGUF
#   Text encoder       BF16 on disk     ~16.0 GB   |
#   VAE                                  ~1.2 GB   | Qwen/Qwen-Image-Edit-2511
#   Configs / scheduler / tokenizer      ~0.1 GB   |
#   ─────────────────────────────────────────────
#   Total download                       ~32 GB
#
# Runtime RAM (after torchao INT4 compresses text encoder from 16 GB → 3.5 GB):
#   GGUF transformer   ~15 GB
#   Text encoder INT4   ~3.5 GB
#   VAE                 ~1.2 GB
#   macOS + overhead    ~6–8 GB
#   ─────────────────────────────
#   Total runtime      ~26 GB  ←  fits in 32 GB
set -euo pipefail

# Load HF_TOKEN from .env if present and not already set
if [[ -f ".env" && -z "${HF_TOKEN:-}" ]]; then
  # shellcheck disable=SC2046
  export $(grep -E '^HF_TOKEN=' .env | xargs)
fi

PYTHON="${PYTHON:-python3.11}"
VENV_DIR="venv"
DATA_DIR="./data"
GGUF_REPO="unsloth/Qwen-Image-Edit-2511-GGUF"
GGUF_FILE="qwen-image-edit-2511-Q5_K_M.gguf"
PIPELINE_REPO="Qwen/Qwen-Image-Edit-2511"
PIPELINE_DIR="${DATA_DIR}/Qwen-Image-Edit-2511"

# ── 1. Python 3.11 check ──────────────────────────────────────────────────────
echo "==> Checking Python version..."
PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYVER" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required, found Python $PYVER"
  echo "       brew install python@3.11"
  exit 1
fi
echo "    Python $PYVER  ($(which "$PYTHON"))"

# ── 2. Virtual environment ─────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating virtual environment at ./$VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "==> Virtual environment already exists — skipping."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "    $(python --version)  |  $(which python)"

# ── 3. Pip + PyTorch ──────────────────────────────────────────────────────────
echo "==> Upgrading pip..."
pip install --quiet --upgrade pip

echo "==> Installing PyTorch (MPS built-in on macOS)..."
pip install --quiet torch torchvision torchaudio

python - <<'PYEOF'
import torch
if torch.backends.mps.is_available():
    print("    MPS backend: available")
else:
    print("    WARNING: MPS not available — inference will run on CPU")
PYEOF

# ── 4. Python dependencies ────────────────────────────────────────────────────
echo "==> Installing dependencies from requirements.txt..."
pip install --quiet -r requirements.txt

# ── 5a. GGUF transformer (~15 GB) ─────────────────────────────────────────────
mkdir -p "$DATA_DIR"
GGUF_LOCAL="${DATA_DIR}/${GGUF_FILE}"

if [[ -f "$GGUF_LOCAL" ]]; then
  echo "==> GGUF already downloaded: $GGUF_LOCAL — skipping."
else
  echo "==> Downloading GGUF transformer: ${GGUF_FILE} (~15 GB)..."
  python - <<PYEOF
import os, sys
from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN") or None
path = hf_hub_download(
    repo_id="$GGUF_REPO",
    filename="$GGUF_FILE",
    local_dir="$DATA_DIR",
    token=token,
)
print(f"    Saved: {path}")
PYEOF
fi

# ── 5b. Pipeline components: text encoder + VAE + configs (~17 GB) ───────────
if [[ -f "${PIPELINE_DIR}/model_index.json" ]]; then
  echo "==> Pipeline components already downloaded: $PIPELINE_DIR — skipping."
else
  echo "==> Downloading pipeline components (~17 GB — text encoder + VAE + configs)..."
  echo "    Skipping 40+ GB transformer safetensors (using GGUF instead)."
  python - <<PYEOF
import os, sys
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
path = snapshot_download(
    repo_id="$PIPELINE_REPO",
    local_dir="$PIPELINE_DIR",
    token=token,
    # Skip transformer weight shards — we use the GGUF file instead
    ignore_patterns=[
        "transformer/diffusion_pytorch_model*.safetensors",
        "transformer/*.bin",
        "*.msgpack",
        "*.h5",
        "flax_model*",
    ],
)
print(f"    Saved: {path}")
PYEOF
fi

# ── 6. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "==> Setup complete!"
echo ""
echo "    Activate env  : source $VENV_DIR/bin/activate"
echo "    CLI inference : python infer.py input.jpg \"make the sky orange\""
echo "    Web app       : python app.py   →  http://localhost:8000"
echo ""
printf "    Data size: "; du -sh "$DATA_DIR" 2>/dev/null | cut -f1
echo "    Tip: close other apps before running — model uses ~26 GB RAM."
