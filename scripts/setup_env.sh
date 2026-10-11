#!/usr/bin/env bash
# GPU environment of the paper: Python 3.12, CUDA 12.8, one NVIDIA A100-SXM4-80GB.
set -euo pipefail
WORK="${JIT_WORK:-/workspace}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
python3 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -m pip install -r "$REPO/requirements-gpu.txt"
python -m pip install -e "$REPO"
python -c "import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__, torch.cuda.get_device_name(0))"
