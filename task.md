# Task List: Paradom Phase 1 Core Experiment

- [x] Upgrade `TinyMamba` architecture with functional SSM scan
    - [x] `paradom/models/tiny_transformer.py`
    - [x] `paradom/models/tiny_mamba.py`
- [x] Implement training infrastructure & Fixes
    - [x] `scripts/data_utils.py` (WikiText subset)
    - [x] `scripts/train_poc.py` (checkpoints, seed, 10,000 samples, 5 epochs)
    - [x] `scripts/experiment_001.py` (directory creation, full report)
- [x] Refine Paradom Core for PoC
    - [x] Extend `FunctionalRoleMatcher` roles
    - [x] `TinyTransformerToMambaMapper` with SSM derivation (A_log, D, conv1d)
    - [x] Real CKA scoring + explicit layer map per SPEC §5.1
- [x] Execute Experiment (WSL Ubuntu)
    - [x] WSL Ubuntu environment
    - [x] Train Transformer (Source) — checkpoints in `checkpoints/`
    - [x] Train Mamba (Target baseline)
    - [x] Run Paradom swap
    - [x] Evaluate results — `research/EXPERIMENT_001_RESULTS.md`
    - [x] Ratio sweep — `research/EXPERIMENT_002_RATIO_SWEEP.md`

## WSL quick run (deps already installed)

```bash
cd /mnt/c/Users/Latera/Desktop/AWFE_Documentation/AWFE
export PYTHONPATH="$PWD"
bash scripts/run_phase1_wsl.sh
```

## Phase 1 verdict (WSL run 2026-07-03)

| Metric | Value |
|---|---|
| Best swap fraction | **20%** |
| Swapped perplexity | **43,971** |
| Intelligence retention | **~67%** |
| vs random init | Pass |
| vs trained Mamba (within 20%) | Not yet — needs Phase 2 scale |
