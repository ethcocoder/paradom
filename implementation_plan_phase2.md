# Paradom Phase 2: Scaling the Swap Engine (Month 4 Plan)

Now that Phase 1 has proven the core concept of weight equivalence and SSM derivation, Phase 2 focuses on scaling these techniques to production-grade LLMs (7B–70B parameters).

## User Review Required

> [!IMPORTANT]
> **External Benchmarks**: Full evaluation of 7B models will require Google Colab or similar environments with high RAM/VRAM.
> **Dependency Update**: We will need `scikit-learn` for its optimized `randomized_svd` implementation to handle 4096+ dimension matrices.

## Proposed Changes

### Core Optimization (Speed & RAM)

#### [MODIFY] [importance.py](file:///c:/Users/Latera/Desktop/AWFE_Documentation/AWFE/paradom/core/importance.py)
- Implement `ImportanceScorer.score_randomized_svd()`: Use `sklearn.utils.extmath.randomized_svd` for $O(N^2)$ instead of $O(N^3)$ complexity on large tensors.

#### [MODIFY] [engine.py](file:///c:/Users/Latera/Desktop/AWFE_Documentation/AWFE/paradom/core/engine.py)
- Refine `Paradom.swap` to support streaming mode more robustly.
- Implement progress tracking for multi-gigabyte weight files.

### 7B Model Support

#### [NEW] [transformer_to_mamba.py](file:///c:/Users/Latera/Desktop/AWFE_Documentation/AWFE/paradom/mappings/transformer_to_mamba.py)
- Implement the "Full Scale" version of the mapping logic.
- Support RoPE (Rotary Positional Embeddings) and GQA (Grouped Query Attention) common in LLaMA 3 and Mistral.

#### [NEW] [configs/mamba_7b.yaml](file:///c:/Users/Latera/Desktop/AWFE_Documentation/AWFE/configs/mamba_7b.yaml)
- Target architecture configuration for a Mamba-7B model compatible with LLaMA 3 8B.

## Open Questions

- Should we implement the associative parallel scan in Python for Phase 2, or rely on existing libraries like `mamba-ssm`? (Propose: use Python/Numpy for the swap engine's verification, but support `safetensors` for loading into `mamba-ssm`).

## Verification Plan

### Automated Tests
1. Benchmark `randomized_svd` vs `torch.linalg.svd` on a $(4096, 4096)$ tensor.
2. Verify streaming RAM usage stays under 4GB when "processing" a fake 7B model.

### Manual Verification
1. Run `paradom identify` on a real HuggingFace model (e.g., `meta-llama/Meta-Llama-3-8B`) and verify the equivalence map is complete.
