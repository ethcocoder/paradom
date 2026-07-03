#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/Latera/Desktop/AWFE_Documentation/AWFE"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY=python3

echo "=== Environment ==="
$PY --version
$PY -c "import torch, safetensors, transformers; print('torch', torch.__version__); print('deps OK')"
$PY -c "from paradom.core.engine import Paradom; print('paradom OK')"

if $PY -c "import pytest" 2>/dev/null; then
  echo ""
  echo "=== Tests ==="
  $PY -m pytest tests/ -v --tb=short
else
  echo ""
  echo "=== Tests skipped (pytest not installed) ==="
fi

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
