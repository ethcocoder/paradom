#!/bin/bash
# AWFE Test B — Colab Runner
# ===========================
# Paste each section into a Colab cell, or run all at once.

# ── Cell 1: Clone & Setup ──
git clone -b v3 https://github.com/ethcocoder/paradom.git
cd paradom
pip install torch transformers -q

# ── Cell 2: Run Tests ──
PYTHONPATH=. python scratch/hf_end_to_end_test.py 2>&1
