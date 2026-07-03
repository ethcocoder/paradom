#!/usr/bin/env bash
# Quick re-run of Experiment 001 only (WSL, user deps already installed)
set -euo pipefail
ROOT="/mnt/c/Users/Latera/Desktop/AWFE_Documentation/AWFE"
cd "$ROOT"
export PYTHONPATH="$ROOT"
python3 scripts/experiment_001.py
cat research/EXPERIMENT_001_RESULTS.md
