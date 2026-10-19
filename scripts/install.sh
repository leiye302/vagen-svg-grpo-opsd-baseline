#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python3.12}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install 'uv==0.9.3' 'setuptools==79.0.1' wheel packaging ninja
.venv/bin/uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0+cu128'
.venv/bin/uv pip install --python .venv/bin/python \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -c constraints.txt -r requirements.txt
.venv/bin/python scripts/install_flash.py
.venv/bin/python -m pip check
.venv/bin/python scripts/bootstrap_sources.py
printf '\nInstalled. Activate with: source .venv/bin/activate\n'
