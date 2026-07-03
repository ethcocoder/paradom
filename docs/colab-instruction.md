# Paradom Phase 1: Google Colab Instructions

Since the Phase 1 goal involves training small models and running weight swaps, you can use Google Colab for a faster execution with T4 GPUs.

## 1. Setup & Clone

Run this in a Colab cell to clone the repository and switch to the `v2` branch:

```bash
# Clone the repository
!git clone https://github.com/ethcocoder/paradom.git
%cd paradom

# Switch to the v2 branch (contains the updated TinyMamba architecture)
!git checkout v2

# Install dependencies
!pip install -e .
!pip install transformers datasets tqdm safetensors
```

## 2. Run the Core Experiment

You can run the full Phase 1 pipeline (Training -> Swap -> Evaluation) with a single command:

```python
# Force retraining with the updated architecture and 10k samples
!rm -rf checkpoints/*
!bash scripts/run_phase1_wsl.sh
```

*(Note: `run_phase1_wsl.sh` works in Colab's Linux environment just like in WSL.)*

## 3. Manual Run (Step-by-Step)

If you want to run steps individually:

### A. Train the models (Transformer & Mamba)
```python
!python scripts/train_poc.py
```

### B. Execute Paradom Swap
```python
!python scripts/experiment_001.py
```

### C. Run Ratio Sweep
```python
!python scripts/experiment_002_ratio_sweep.py
```

## 4. Downloading Results
After running, your results will be in the `research/` and `output/` folders. You can download the final report and checkpoints:

```python
from google.colab import files
files.download('research/EXPERIMENT_001_RESULTS.md')
files.download('research/EXPERIMENT_001_RESULTS.json')
```
