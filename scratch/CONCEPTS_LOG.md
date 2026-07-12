# AWFE Test B — Exploration Concepts & Results Log

## Problem Statement
Test B: Project SmolLM-135M (576 hidden, 9 heads, 3 kv_heads) → (512 hidden, 8 heads, 2 kv_heads)
Goal: Swapped model produces coherent text output.

## Current Status
- Tests A & C: PASS (identical/coherent output)
- Test B: FAIL (semi-coherent → garbage with varied tokens)

---

## Experiments Tried

### 1. Flat SVD Projection (baseline)
- `_projected_swap` in swap_engine.py
- Projects (192, 576) → (128, 512) as a flat matrix via SVD
- CKA: 0.9990 | E_ratio: ~0.58 (k_proj) → garbage output
- **Finding**: CKA measures structural similarity but NOT functional quality

### 2. CKA Denominator Bug Fix
- File: paradom/core/cka.py:32
- Changed `denom = hsic_xx * hsic_yy` → `denom = (hsic_xx * hsic_yy).sqrt()`
- Correct CKA formula: `HSIC(X,Y)/sqrt(HSIC(X,X)*HSIC(Y,Y))`

### 3. Head-Aware Projection (DISABLED)
- `_projected_swap_head_aware` in swap_engine.py
- Three approaches tried:
  a) Head dropping by Frobenius norm: CKA=0.9174, gibberish
  b) Head merging by cosine similarity: CKA=0.8635, gibberish
  c) Project-all-then-select-heads: CKA=0.9132, gibberish
- **Finding**: Head-aware methods had LOWER CKA than flat SVD (0.91 vs 1.0)
- **Finding**: Cosine similarity of raw weight vectors ≠ functional similarity
- **Finding**: Flat SVD is mathematically optimal for single-step projection

### 4. Energy Rescaling (APPLIED ✅)
- Added Frobenius-norm energy preservation to `_projected_swap`
- Scale: `(src_energy/proj_energy).sqrt().clamp(1.0, 2.0)`
- Result: All E_ratio = 1.0000, but output still not coherent
- **Finding**: Energy preservation is necessary but not sufficient
- **Finding**: The issue is structural (head mixing), not just magnitude

### 5. PCA-Based Activation Projection (WORSE ❌)
- Used source model activations to compute PCA basis
- PCA finds top-128 subspace of actual k/v activations
- Result: CKA dropped to 0.8174, output WORSE than flat SVD
- **Finding**: PCA on activations removes head structure information
- **Finding**: PCA optimizes for variance preservation, not attention quality

### 6. Activation-Aware Head Merging (APPLIED ✅)
- Implemented in `paradom/core/activation_aware_projector.py`
- Uses calibration data to merge similar heads weighted by importance
- Greedy merging of most similar pair (optimal for 3→2 kv_heads)
- Result: CKA=0.9200, overlap=20.83% (vs SVD: CKA=0.9995, overlap=18.75%)
- **Finding**: Lower CKA but slightly better output — confirms CKA is misleading
- **Finding**: Marginal improvement — damage is distributed across ALL projections

### 7. Ablation Test Results
- No single weight category is the bottleneck
- Replacing ANY category with source originals barely helps
- Best ablation: layer_1 at 22.9% (+4.2%)
- **Conclusion**: ALL 273 projections are slightly wrong, errors compound through 30 layers
- **Conclusion**: Fixing individual weight projections won't solve the problem

### 8. Layer-by-Layer Output Alignment (IMPLEMENTED, NOT YET TESTED)
- Implemented in `paradom/core/layer_aligner.py`
- Approach: After weight projection, align each layer's output using Procrustes correction
- Absorbs correction into o_proj (no extra parameters)
- Two modes:
  - `align()`: Source-only calibration (fast, approximate)
  - `align_with_target()`: Full calibration with target model (slow, accurate)
- References: THESEUS (ICML 2026), CAST (2025)
- **Status**: Implemented, needs testing on Colab

---

## Root Cause Analysis

The core problem: k_proj goes from (192, 576) → (128, 512)
- Row reduction: 192 → 128 (33% reduction, 3 kv_heads → 2 kv_heads)
- Col reduction: 576 → 512 (11% reduction)
- SVD loses 42% of energy in the row reduction

The SVD treats the matrix as flat, so the top-128 singular vectors
mix information across all 3 source heads. The target model expects
clean head structure (64 dims per head), but gets linear combinations.

### Architecture Mismatch
- Source: 9 q_heads / 3 kv_heads = GQA 3:1
- Target: 8 q_heads / 2 kv_heads = GQA 4:1
- The attention grouping changes, making exact equivalence impossible

---

## Concepts for Future Exploration

### A. Gradual Multi-Step Reduction
Instead of one big jump (576→512), reduce in many small steps:
- 576 → 568 → 560 → ... → 512 (each step ~1.4% reduction)
- Each step is nearly lossless
- **Status**: Not yet implemented

### B. Parameter/Checkpoint Splitting (user idea)
Split 130M model into two ~65M halves, project each independently.
Possible interpretations:
1. **Split by layers**: First 15 layers → project, last 15 layers → project
   - Problem: layers are sequential, not independent
   - The second half depends on the first half's output
2. **Split d_model**: 576 = 288 + 288, project each half
   - Each half has smaller reduction ratio
   - But total information loss is the same
3. **Cascaded projection**: Project layer-by-layer, using already-projected
   layers' activations as context for the next layer's projection
   - Most principled approach
   - Requires running partial model during projection

### C. Architecture-Aware Simplification
"Replace hardest maths with smaller maths":
- Keep the same function (attention) but with fewer parallel computations
- 3 KV heads → 2 KV heads = same knowledge, simpler formula
- Need to understand WHAT each head computes, not just its weights

### D. Head Merging by Functional Contribution
Instead of weight similarity, measure each head's contribution to
the attention OUTPUT:
1. Run source model, capture per-head attention outputs
2. Compute which 2 heads best approximate the full 3-head output
3. Construct projected weights to preserve those 2 heads

### E. GQA-Aware Projection
Account for the GQA ratio change (3:1 → 4:1):
- Source: each kv_head serves 3 q_heads
- Target: each kv_head serves 4 q_heads
- The projection should optimize for the target's GQA grouping

---

## Key Files
- `paradom/core/swap_engine.py`: Projection methods (SVD, head-aware, PCA)
- `paradom/mappings/transformer_to_transformer.py`: Mapper with PCA routing
- `scratch/hf_end_to_end_test.py`: Main test (A, B, C)
- `scratch/debug_test_b.py`: Deep diagnostics
- `paradom/core/cka.py`: CKA computation (sqrt fix applied)

## Run Commands
```bash
wsl bash -c "export PYTHONPATH=.:$PYTHONPATH && .venv-wsl/bin/python3 scratch/hf_end_to_end_test.py 2>&1"
wsl bash -c "export PYTHONPATH=.:$PYTHONPATH && .venv-wsl/bin/python3 scratch/debug_test_b.py 2>&1"
```
