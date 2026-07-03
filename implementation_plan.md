# Implementation Plan: Paradom Phase 1 Core Experiment (Execution & WSL)

This plan outlines the steps to prove weight equivalence by swapping parameters between a 2-layer TinyTransformer and a 2-layer TinyMamba.

## User Review Required

### WSL Post-Reboot Steps
> [!IMPORTANT]
> Since you are about to restart, please ensure:
> 1. **Virtualization** is enabled in BIOS/UEFI.
> 2. After reboot, run `wsl --install -d Ubuntu` again.
> 3. Once Ubuntu is running, you'll need to install dependencies:
>    ```bash
>    sudo apt update && sudo apt install python3-pip -y
>    pip install torch transformers safetensors numpy scipy POT huggingface-hub pyyaml typer rich tqdm psutil
>    ```

## Proposed Changes

### 1. Training Infrastructure & Script Fixes

#### [MODIFY] [train_poc.py](file:///c:/Users/Latera/Desktop/AWFE_Documentation/AWFE/scripts/train_poc.py)
- Delete redundant import at line 7 and ensure `from paradom.models.tiny_mamba import TinyMamba` is used.
- Add `os.makedirs("checkpoints", exist_ok=True)`.
- Change save paths to `checkpoints/tiny_transformer_trained.pt` etc.

#### [MODIFY] [experiment_001.py](file:///c:/Users/Latera/Desktop/AWFE_Documentation/AWFE/scripts/experiment_001.py)
- Add directory safety: `os.makedirs("research", exist_ok=True)` and `os.makedirs("output", exist_ok=True)`.
- Update paths to point to the `checkpoints/` folder.

### 2. Paradom Core Refinement

#### [MODIFY] [engine.py](file:///c:/Users/Latera/Desktop/AWFE_Documentation/AWFE/paradom/core/engine.py)
- Refine the `identify` method's `target_requirements` to match the exact layer names and shapes in `TinyMamba` (specifically for `in_proj`, `x_proj`, `out_proj`, and `dt_proj`).
- Ensure `architecture="tinytransformer"` is correctly handled in `stream_layers`.

## Open Questions

- **Dataset Size**: The current `train_poc.py` uses `num_samples=2000`. Is this sufficient for a quick proof of concept, or should we increase it once in WSL? (Proposed: Keep at 2k for the first run to verify the pipeline).

## Verification Plan

### Automated Tests (To be run in WSL)
1. `python3 scripts/train_poc.py` (Verify models train and save).
2. `python3 scripts/experiment_001.py` (Verify swap logic and result generation).

### Manual Verification
- Review `research/EXPERIMENT_001_RESULTS.md`.
- Success metric: **Swapped Mamba Perplexity < (Random Mamba Perplexity - 10%)**.
