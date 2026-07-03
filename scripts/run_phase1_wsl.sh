#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY=python3

echo "=== Environment ==="
$PY --version
# Force editable install to refresh package structure in the current environment
$PY -m pip install -e . --quiet
$PY -c "import torch, safetensors, transformers; print('torch', torch.__version__); print('deps OK')"
$PY -c "import paradom; print('paradom module path:', paradom.__file__)"
$PY -c "from paradom.core.engine import Paradom; print('paradom core OK')"



if [[ ! -f checkpoints/tiny_transformer_trained.pt ]] || [[ ! -f checkpoints/tiny_mamba_trained.pt ]]; then
  echo ""
  echo "=== Training ==="
  $PY scripts/train_poc.py
else
  echo ""
  echo "=== Checkpoints found — skipping training ==="
fi

echo ""
echo "=== Experiment 001 ==="
$PY scripts/experiment_001.py

echo ""
echo "=== Experiment 002 ==="
$PY scripts/experiment_002_ratio_sweep.py

echo ""
echo "=== Results ==="
cat research/EXPERIMENT_001_RESULTS.md
echo ""
echo "--- Ratio sweep best ---"
$PY -c "import json; d=json.load(open('research/EXPERIMENT_002_RATIO_SWEEP.json')); print('best', d['best'])"
