# Roadmap: Language-Aware Dimension Truncation

**Paradom Experiment 003+**
**Date:** 2026-07-12
**Problem:** SmolLM-135M downscales 576→512 (d_model), 1536→1408 (FFN), 9→8 Q heads, 3→2 KV heads. Current CKA=0.5577, output degenerates to "WISEWISE..." repetition loop.
**Hypothesis:** Uniform SVD truncation blends all semantic information across the retained space, destroying coherent subspaces. Instead, identify whole semantic clusters ("languages"), rank by importance, and preserve the top ones intact while dropping less important clusters wholesale.

---

## Core Principle

```
Current approach (SVD spectral truncation):
  576-dim vector → PCA → keep top 512 components
  Problem: every retained component is a mix of ALL original features.
           A "word" that lived in dimension 400 is now smeared across
           all 512 retained dimensions, partially preserved everywhere,
           fully preserved nowhere.

Language-aware approach:
  576-dim vector → identify 16 "semantic languages" of 36 dims each
  Rank languages by activation variance / gradient sensitivity
  Keep top 14 languages (14 × 36 = 512 dims) IN FULL
  Drop bottom 2 languages (2 × 36 = 72 dims) IN FULL
  Result: 14 languages are 100% preserved, 0 are 0% preserved.
  No language is 70% preserved → no language is destroyed.
```

The head-aware attention swap already does exactly this for heads: it keeps whole heads (via Frobenius norm ranking) rather than blending across head boundaries. This roadmap extends that same philosophy to d_model, FFN inner dim, and embeddings.

---

## Phase 1: Dimension Language Discovery (Calibration-Free)

**Goal:** Without any calibration data, identify natural "language" clusters in weight matrices and rank them by structural importance.

**Duration:** ~1 week
**Depends on:** Nothing (pure tensor analysis)

### 1.1 Define "Language" Structure

For a weight matrix W ∈ R^(m × n), a "language" is a contiguous or structured block of columns (or rows) that form a coherent subspace.

**Three candidate language definitions:**

| Definition | Structure | When to use |
|---|---|---|
| **Block languages** | Consecutive slices: dims [0:36], [36:72], ... [540:576] | Default. Simple, fast, matches head_dim structure. |
| **SVD-component languages** | Each singular vector direction is a "language" | When the semantic structure is aligned with principal components (may mix features) |
| **Activation-cluster languages** | K-means on column-norm vectors of W, group similar columns | When dimensions don't have natural block structure |

**Decision:** Start with **block languages** where block size = head_dim (576 / 9 heads = 64 per head). This gives us 9 d_model-blocks that align exactly with the existing attention head structure. Then generalize.

### 1.2 Build `DimensionLanguageRanker`

New class in `paradom/core/importance.py`:

```
DimensionLanguageRanker
├── rank_by_frobenius(weight_matrix, block_size) → sorted indices
├── rank_by_activation_variance(weight_matrix, activations) → sorted indices  
├── rank_by_gradient_sensitivity(weight_matrix, gradient_data) → sorted indices
└── select_top_k(weight_matrix, target_dim, ranking_method) → truncated_weight, mask
```

**Ranking methods (from cheapest to most informative):**

1. **Frobenius per-block** (no calibration needed): For each block of `block_size` columns, compute ||W[:, block]||_F. Rank blocks by this norm. Keep top k. *This is exactly what head-aware already does for attention, extended to arbitrary blocks.*

2. **Activation variance** (needs 1 forward pass): Run calibration data through source model. For each dimension j, compute Var(activations[:, j]) across the batch. Dimensions with high variance are "active" — they discriminate between inputs. Keep top k blocks by mean variance.

3. **Gradient sensitivity** (needs forward + backward): Compute ∂L/∂W per-block. High gradient = high importance. Most expensive, most accurate.

### 1.3 Milestones

| Milestone | Deliverable | Success Criteria |
|---|---|---|
| M1.1 | `DimensionLanguageRanker` class with all 3 methods | Unit tests pass, rank_by_frobenius matches existing head norms for attention weights |
| M1.2 | Block-size auto-detection: given d_model and n_heads, infer block_size = d_model / n_heads | Correctly infers 64 for SmolLM (576/9) |
| M1.3 | `select_top_k` produces correct truncated weight + consistent column mask | Truncated weight shape is (m, target_dim), mask is boolean (m, src_dim) with exactly target_dim True values per row-group |

### 1.4 Validation Experiment

Run on SmolLM-135M with block_size=64:
- Compare CKA of block-truncated vs SVD-truncated for the same d_model reduction (576→512)
- Block truncation drops 1 whole "language" (64 dims). SVD truncation keeps top 512 components.
- **Hypothesis:** Block truncation will have lower per-layer CKA but *higher* end-to-end output quality because it preserves coherent subspaces.

---

## Phase 2: Cross-Layer Consistent Language Mapping

**Goal:** When d_model changes at one layer, the SAME dimensions must be dropped at every layer so that layer i's output space matches layer i+1's input space.

**Duration:** ~1 week
**Depends on:** Phase 1

### 2.1 Shared Language Mask

The critical insight: if dim 400-464 is dropped at layer 0, it must also be dropped at layers 1-29. Otherwise layer 0 outputs 512 dims but layer 1 expects 576 dims as input → shape mismatch or silent corruption.

**Implementation:**
- Compute the language ranking ONCE from a reference weight (e.g., the embedding matrix, or layer 0's Q projection)
- Store the global `keep_mask: Tensor[576]` (boolean, 512 True / 64 False)
- Apply this SAME mask when reducing every weight matrix's d_model dimension

This already exists partially: `_get_spectral_projector` in `swap_engine.py:145` caches projectors by `(axis_label, direction)`. The language-aware approach replaces the SVD projector with a column-selection mask, but the caching pattern is identical.

### 2.2 Bidirectional Consistency

For a transformer layer:
```
Input (576) → Q/K/V projections → Attention → O projection (576) → Residual → FFN
                                                              ↓
                                                    gate/up proj (1536)
                                                    down proj (576)
```

Two distinct reduction points:
1. **d_model reduction** (576→512): Affects embeddings, all Q/K/V/O projections, FFN down_proj, layer norms, residual stream
2. **FFN inner dim reduction** (1536→1408): Affects gate_proj, up_proj, down_proj

Each needs its own consistent language mask. They are independent — the FFN inner dim languages don't need to align with d_model languages.

### 2.3 The Consistency Protocol

```
1. Load source model
2. Compute d_model language ranking from embedding weight (most informative — 
   it sees all tokens)
3. Compute FFN inner dim language ranking from layer 0's gate_proj (first FFN layer)
4. Freeze both rankings as global masks
5. For every layer:
   - Apply d_model mask to: Q, K, V, O projections, layer norms, residual adds
   - Apply FFN mask to: gate_proj, up_proj inner dims; down_proj output dim
   - No layer computes its own ranking — all use the global masks
```

### 2.4 Milestones

| Milestone | Deliverable | Success Criteria |
|---|---|---|
| M2.1 | `SharedLanguageMask` class that computes ranking once and applies everywhere | All 30 layers of SmolLM use identical column selection |
| M2.2 | Integration with `_ot_swap` replace SVD projector with language mask projector | Output shapes are correct for all weight matrices |
| M2.3 | End-to-end swap of SmolLM-135M with consistent masking | No shape mismatches, all 30 layers process successfully |

### 2.5 Validation Experiment

- Swap SmolLM with consistent language masking
- Measure per-layer CKA (should be lower than SVD per-layer, since we're dropping whole blocks)
- Measure output quality (perplexity, generation coherence)
- **Hypothesis:** Per-layer CKA drops but output quality improves because the retained subspaces are internally coherent

---

## Phase 3: Calibration-Informed Language Ranking

**Goal:** Use actual model activations on real data to determine which dimensions are semantically important, rather than relying on weight magnitudes alone.

**Duration:** ~1.5 weeks
**Depends on:** Phase 2

### 3.1 Calibration Data Pipeline

```
1. Load SmolLM-135M source model
2. Run 1000 calibration sentences through it
3. Capture per-layer activations: shape (batch, seq_len, d_model) for each layer
4. Compute per-dimension statistics:
   - Variance across batch+seq: high variance = discriminative dimension
   - Mean absolute activation: high magnitude = active dimension
   - Mutual information with next-token prediction: most direct importance signal
5. Aggregate across layers: a dimension that's important at layer 0 AND layer 15 
   is more important than one important only at layer 0
```

### 3.2 Multi-Signal Ranking

Combine signals into a single importance score per dimension:

```
importance(d) = α · normalized_variance(d) 
              + β · normalized_magnitude(d) 
              + γ · normalized_gradient(d)
```

Where α, β, γ are hyperparameters (start with 0.5, 0.3, 0.2).

For the block language approach, aggregate within each block:
```
block_importance(block) = mean(importance(d) for d in block)
```

### 3.3 Adaptive Block Size

Not all languages need to be the same size. If the model has:
- 3 attention heads that each use 64 dims → 3 blocks of 64
- Residual stream dimensions that don't align with heads → blocks of 16 or 32

Use hierarchical clustering on the column-norm vectors of the embedding matrix to discover natural block boundaries, rather than imposing a fixed block size.

### 3.4 Milestones

| Milestone | Deliverable | Success Criteria |
|---|---|---|
| M3.1 | Calibration pipeline: run source model, extract per-dim statistics | Produces importance scores for all 576 dimensions |
| M3.2 | Multi-signal ranking with configurable weights | Rankings differ meaningfully from pure Frobenius (validates that calibration adds information) |
| M3.3 | Adaptive block discovery via hierarchical clustering | Produces variable-size blocks that align with known head structure |
| M3.4 | SmolLM swap using calibration-informed ranking | CKA > 0.60 AND generation is coherent (no repetition loops) |

### 3.5 Validation Experiment

- Compare three rankings on SmolLM: (a) Frobenius-only, (b) calibration-informed, (c) SVD baseline
- Measure: CKA, perplexity on WikiText-2, generation coherence (manual + automated repetition detection)
- **Success criterion:** Calibration-informed ranking produces strictly better perplexity than Frobenius-only, and both beat SVD

---

## Phase 4: FFN Language-Aware Truncation

**Goal:** Apply the same "keep whole languages" principle to the FFN layers (gate_proj, up_proj, down_proj), where the reduction is 1536→1408 (drop 128 inner dimensions).

**Duration:** ~1 week
**Depends on:** Phase 2

### 4.1 FFN Language Definition

FFN inner dimensions don't have the natural head-based structure that attention has. Two approaches:

**Approach A: Fixed blocks**
- 1536 / 128 = 12 blocks of 128 dims each
- Drop 1 block to go from 1536→1408
- Simple, but 128-dim blocks may be too coarse or too fine

**Approach B: Neuron Importance**
- Each FFN neuron (row of gate_proj, column of down_proj) is an independent unit
- Rank neurons by: ||gate_proj[i, :]|| · ||up_proj[i, :]|| · ||down_proj[:, i]|| (product of norms across all three matrices that touch neuron i)
- Keep top 1408 neurons, drop bottom 128
- This is a "block language" where each block is size 1 (one neuron)

**Approach C: Clustered neurons**
- K-means on the rows of gate_proj to find 12 clusters of ~128 neurons each
- Rank clusters by mean norm, keep top 11 clusters (1408 neurons)
- Neurons in the same cluster share similar input patterns → they form a coherent "language"

### 4.2 FFN Consistency

gate_proj and up_proj both map d_model → d_inner. They must share the same neuron ordering (same language assignment). down_proj maps d_inner → d_model, so it must use the same neuron ranking on its input dimension.

Implementation: compute neuron ranking from gate_proj (largest FFN matrix), apply same ranking to up_proj rows and down_proj columns.

### 4.3 Milestones

| Milestone | Deliverable | Success Criteria |
|---|---|---|
| M4.1 | `FFNLanguageRanker` implementing approaches A, B, C | All three produce valid neuron/block rankings |
| M4.2 | Integration into `TransformerToTransformerMapper.convert()` | FFN layers use language-aware truncation instead of OT/SVD |
| M4.3 | SmolLM swap with FFN language-aware truncation | gate_proj output is (1408, 512), down_proj output is (512, 1408), shapes consistent |

---

## Phase 5: Embedding Language-Aware Truncation

**Goal:** Truncate the embedding table (vocab_size × 576 → vocab_size × 512) by dropping 64 embedding dimensions, using the same global language ranking from Phase 2.

**Duration:** ~3 days
**Depends on:** Phase 2

### 5.1 Embedding as Language Source

The embedding matrix is the BEST source for language ranking because:
1. It's the largest weight matrix (50257 × 576 for SmolLM)
2. Every token has an embedding → the statistics are most representative
3. It feeds into every layer → its language structure propagates everywhere

**Method:** Use the embedding matrix itself to compute the d_model language ranking. The global keep_mask from Phase 2 should be DERIVED from the embedding, not from any single layer.

### 5.2 Embedding + LM Head Consistency

The lm_head is typically the embedding transposed (weight tying). If we drop embedding dims, lm_head must drop the same dims on its output dimension.

### 5.3 Milestones

| Milestone | Deliverable | Success Criteria |
|---|---|---|
| M5.1 | Embedding language ranking derived from embedding matrix | Rankings align with (or improve upon) attention-derived rankings |
| M5.2 | lm_head uses same mask as embedding | Weight tying preserved: lm_head = embedding.T after truncation |

---

## Phase 6: Unified Pipeline & Experiment 003

**Goal:** Integrate all phases into a single swap pipeline and run the definitive benchmark.

**Duration:** ~1 week
**Depends on:** Phases 1-5

### 6.1 Pipeline Integration

The new `TransformerToTransformerMapper.convert()` flow:

```
1. Load source model weights
2. Compute GLOBAL language rankings:
   - d_model ranking from embedding matrix (Phase 5)
   - FFN inner ranking from layer 0 gate_proj (Phase 4)
3. Freeze rankings as consistent masks
4. For each layer (0 to 29):
   a. Apply d_model mask to: Q/K/V/O projections, layer norms
   b. Apply FFN mask to: gate_proj, up_proj, down_proj
   c. Head-aware truncation for Q (9→8) and KV (3→2) — unchanged from current
5. Apply d_model mask to: embedding, lm_head, final norm
6. Compute CKA, save output
```

### 6.2 Experiment 003: Language-Aware vs SVD Baseline

**Comparison matrix:**

| Method | d_model truncation | FFN truncation | Head truncation | Expected CKA | Expected coherence |
|---|---|---|---|---|---|
| Current (SVD) | Spectral projection | OT/SVD | Head-aware (already good) | 0.557 | "WISEWISE..." |
| Block languages (fixed) | Drop 1 block of 64 | Drop 1 block of 128 | Head-aware | ~0.50 | Better — no blending |
| Neuron importance (FFN) | Drop 1 block of 64 | Drop bottom 128 neurons | Head-aware | ~0.52 | Better — neurons preserved whole |
| Calibration-informed | Variance-ranked blocks | Gradient-ranked neurons | Head-aware | ~0.58+ | Best — data-driven |
| Full language-aware (adaptive) | Cluster-based blocks | Cluster-based blocks | Head-aware | ~0.60+ | Best — natural structure |

### 6.3 Evaluation Metrics

| Metric | How to measure | Target |
|---|---|---|
| **CKA (per-layer)** | Existing `weight_cka()` | > 0.60 mean across layers |
| **CKA (end-to-end)** | Propagate both models on same input, CKA between final hidden states | > 0.70 |
| **Perplexity** | WikiText-2 test set | < 30.0 (vs trained baseline ~17, random ~59000) |
| **Generation coherence** | Manual inspection + automated repetition detector | No repetition loops, grammatical fragments |
| **Cross-layer shape consistency** | Assert all intermediate shapes match target config | 100% pass |

### 6.4 Milestones

| Milestone | Deliverable | Success Criteria |
|---|---|---|
| M6.1 | Unified pipeline producing swapped SmolLM checkpoint | All shapes correct, checkpoint loads without error |
| M6.2 | Experiment 003 benchmark results document | Table comparing all methods across all metrics |
| M6.3 | Generation test: run swapped model on prompts | Coherent output, no "WISEWISE..." repetition |
| M6.4 | Updated `ROADMAP.md` and `AUDIT.md` with findings | Honest assessment of what improved and what didn't |

---

## Phase 7: Generalization & Hardening

**Goal:** Make the language-aware approach work for arbitrary model pairs, not just SmolLM.

**Duration:** ~1.5 weeks
**Depends on:** Phase 6

### 7.1 Auto-Detection of Language Structure

Given any source model:
1. Detect d_model, n_heads, head_dim → infer natural block structure
2. Detect FFN inner dim → infer neuron count
3. Auto-select ranking method based on available calibration data
4. If no calibration data → fall back to Frobenius ranking

### 7.2 Non-Transformer Models

Extend language-aware truncation to:
- **Mamba SSM:** d_model reduction applies to in_proj, x_proj, out_proj. The SSM state space (A, B, C matrices) has different structure — "languages" here are SSM state dimensions.
- **MoE:** Each expert is independent → languages can be per-expert
- **Vision models:** CNN channels are natural "languages" (each channel = one feature detector)

### 7.3 Milestones

| Milestone | Deliverable | Success Criteria |
|---|---|---|
| M7.1 | Auto-detection module for arbitrary architectures | Correctly handles SmolLM, LLaMA-3-8B, Mistral-7B configs |
| M7.2 | Mamba language-aware truncation | Mamba swap uses same framework, consistent masking |
| M7.3 | Documentation: "Language-Aware Truncation" section in ARCHITECTURE.md | Clear explanation of the approach, when to use it, limitations |

---

## Risk Register (New Risks from This Approach)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Block boundaries don't align with semantic structure → worse than SVD | Medium | High | Phase 1 validation experiment catches this early. Fall back to SVD if block CKA < SVD CKA. |
| Adaptive block sizes create shape mismatches across layers | Medium | High | Phase 2 consistency protocol enforces uniform blocks. Only allow adaptive within the d_model or FFN dimension, not across. |
| Calibration data overfits to specific prompts → ranking doesn't generalize | Low | Medium | Use diverse calibration set (multiple domains). Compare rankings across different calibration sets — if they agree, ranking is robust. |
| Computational cost of calibration passes exceeds SVD cost | Low | Low | Frobenius ranking is free (just weight norms). Calibration adds one forward pass — negligible for 135M model. |
| Head-aware + language-aware double truncation compounds error | Medium | Medium | Measure: does keeping 8 Q-heads AND dropping 64 d_model dims lose too much? May need to reduce one when increasing the other. |

---

## Dependencies on Existing Code

| Component | File | Change needed |
|---|---|---|
| `ImportanceScorer` | `paradom/core/importance.py:5` | Add `DimensionLanguageRanker` methods |
| `SwapEngine._ot_swap` | `paradom/core/swap_engine.py:100` | Add `language_mask` swap type alongside SVD projector |
| `TransformerToTransformerMapper._apply_swap` | `paradom/mappings/transformer_to_transformer.py:158` | Accept language mask parameter |
| `TransformerToTransformerMapper._head_aware_attention_swap` | `paradom/mappings/transformer_to_transformer.py:187` | No change — already does the right thing |
| `TransformerToTransformerMapper.convert` | `paradom/mappings/transformer_to_transformer.py:22` | Compute global rankings before layer loop |

---

## Priority Order (What to Build First)

If time is limited, build in this order:

1. **Frobenius block ranking + consistent masking** (Phase 1 + Phase 2)
   - This is 80% of the value with 20% of the effort
   - Pure tensor operations, no calibration needed
   - Can validate immediately by comparing CKA

2. **Embedding-derived ranking** (Phase 5)
   - Trivial once Phase 2 exists
   - The embedding is the best single source of language structure

3. **FFN neuron importance** (Phase 4)
   - Straightforward extension
   - Product of norms across gate/up/down is a clean signal

4. **Calibration-informed ranking** (Phase 3)
   - Most work, most uncertain payoff
   - Do this only if Frobenius ranking shows improvement but isn't enough

5. **Adaptive block sizes** (Phase 3.3)
   - Nice-to-have, not essential for first results

---

## Expected Outcome

**Current state:** CKA=0.5577, "WISEWISE..." repetition (model has lost coherent representation structure)

**After Phase 1+2 (block languages + consistency):** CKA may drop slightly (~0.50-0.53 per-layer) because we're dropping whole blocks rather than spreading the loss evenly. But output quality should improve because retained dimensions are internally coherent. **The repetition loop should break.**

**After Phase 3+4+5 (calibration + FFN + embeddings):** CKA should recover to ~0.58-0.62 because calibration data guides us to keep the RIGHT blocks. Perplexity should drop significantly from current levels.

**Success threshold:** Swapped model generates grammatically coherent text (no repetition loops) with perplexity < 25.0 on WikiText-2. This would demonstrate that language-aware truncation preserves semantic structure that SVD destroys.
