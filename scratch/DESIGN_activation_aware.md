# Design: Activation-Aware Weight Projection

## 1. Problem Statement

**Goal**: Project SmolLM-135M (576 hidden, 9 heads, 3 kv_heads) into a smaller LlamaForCausalLM (512 hidden, 8 heads, 2 kv_heads). The swapped model should produce coherent text.

**Current failure**: SVD projection achieves CKA=0.9995 (near-perfect structural similarity) but produces garbage output. The problem weight is k_proj: `(192, 576) -> (128, 512)`.

**Root cause**: SVD minimizes Frobenius norm, but the model's functional quality depends on attention patterns, which depend on head geometry. SVD mixes information across head boundaries, creating linear combinations that the target model's attention mechanism cannot interpret.

---

## 2. Why SVD Fails (Detailed Analysis)

### The Weight Shape Problem

```
Source k_proj: (192, 576) = 3 kv_heads x 64 head_dim, d_model=576
Target k_proj: (128, 512) = 2 kv_heads x 64 head_dim, d_model=512
```

SVD treats this as a flat matrix. The top-128 singular vectors of a (192, 576) matrix create linear combinations of ALL 192 rows. The result is a (128, 576) matrix where each row is a mixture of all 3 source heads.

The target model expects:
- Rows 0-63: clean kv_head_0 (64 dims)
- Rows 64-127: clean kv_head_1 (64 dims)

But SVD gives:
- Rows 0-63: linear combinations of heads 0, 1, 2
- Rows 64-127: different linear combinations of heads 0, 1, 2

### Why This Breaks Attention

The attention mechanism computes:
```
Q = X @ W_q^T   # (T, 512)
K = X @ W_k^T   # (T, 128) -- after projection
```

In the target model with GQA (8 q_heads, 2 kv_heads):
- q_heads 0-3 attend to kv_head 0 (rows 0-63 of K)
- q_heads 4-7 attend to kv_head 1 (rows 64-127 of K)

But after SVD projection, rows 0-63 of K contain a mixture of ALL 3 source heads. The q_heads that attend to these rows get meaningless input because the "head structure" has been destroyed.

### Why CKA Is Misleading

CKA measures:
```
CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
```

This measures overall matrix similarity, not head-structure preservation. A matrix that mixes all heads can still be very close to the original in Frobenius norm (hence high CKA) while being completely wrong functionally (because attention depends on head-level geometry).

### Energy Rescaling Doesn't Help

We tried scaling the projected matrix to preserve total energy:
```python
scale = (src_energy / proj_energy).sqrt()
```
This preserves the overall magnitude but not the head structure. The mixed-head rows still produce meaningless attention patterns.

---

## 3. The Activation-Aware Approach

### Key Insight

Instead of asking "what weight matrix is closest to the source?", we ask: **"what weight matrix produces the same attention behavior?"**

The attention mechanism depends on:
1. **Q-K alignment**: Which positions attend to which (attention patterns)
2. **V aggregation**: What information is extracted at attended positions

Both depend on HEAD GEOMETRY, not overall weight magnitude.

### Algorithm Overview

```
1. CALIBRATE: Run source model on a short prompt
   - Collect Q, K, V activations from each layer
   - These reveal HOW the model uses its heads

2. SCORE HEADS: For each layer, compute:
   - head_importance[h]: How much head h contributes to attention output
   - head_similarity[h1,h2]: How similar are the attention patterns

3. MERGE HEADS: Greedy merging of most similar pair
   - Weight by importance (important heads contribute more)
   - This preserves attention geometry while reducing head count

4. PROJECT D_MODEL: Standard SVD for column reduction
   - No head structure in d_model dimension, so SVD is fine
```

### Why This Works

**Head merging preserves attention geometry**: When two heads have similar attention patterns, their information is redundant. Merging them (weighted average) preserves the combined information while reducing the head count.

**SVD preserves input subspace**: The d_model dimension has no head structure. SVD finds the best low-rank approximation of the input mapping, which is appropriate here.

**Calibration captures real usage**: The attention patterns from calibration data reveal which heads the model actually relies on, not just which heads have large weight norms.

---

## 4. Mathematical Justification

### 4.1 Head Importance

For kv_head h, its importance is measured as the total contribution to the attention output across all q_heads:

```
importance[h] = sum_i ||attn_weighted_v[i,h]||
```

where:
```
attn[i,h] = softmax(q_i @ k_h^T / sqrt(d))     # (T, T)
attn_weighted_v[i,h] = mean_t(attn[i,h] @ v_h)  # (d,)
```

This measures the actual amount of information flowing through kv_head h.

**Mathematical basis**: The attention output for q_head i is:
```
out_i = sum_h attn[i,h] @ v_h
```

The contribution of kv_head h is `attn[i,h] @ v_h`. Its magnitude (Frobenius norm) measures how much information passes through.

### 4.2 Head Similarity

Two kv_heads are "similar" if they contribute similarly to the attention output:

```
similarity[h1,h2] = mean_i cosine(attn_weighted_v[i,h1], attn_weighted_v[i,h2])
```

**Mathematical basis**: If two heads have identical attention-weighted value outputs, merging them is lossless:
```
merged = (w1 * v1 + w2 * v2)  # preserves the combined information
```

### 4.3 Optimal Merging Weights

The merge weights are proportional to importance:
```
w_i = importance[i] / (importance[i] + importance[j])
```

**Justification**: This preserves the energy distribution. The more important head contributes more to the merged result, maintaining the relative scale of attention outputs.

### 4.4 Why Greedy Is Optimal for 3->2

With 3 source kv_heads and 2 target kv_heads, there are exactly 3 possible merge strategies:
1. Merge heads {0,1}, keep head 2
2. Merge heads {0,2}, keep head 1
3. Merge heads {1,2}, keep head 0

The greedy algorithm (merge most similar pair) evaluates all 3 and picks the best. This is globally optimal.

For the 9->8 q_head reduction, there are C(9,2)=36 possible pairs. Greedy still works well because we only remove one head.

---

## 5. Algorithm Pseudocode

### 5.1 Calibration Phase

```
function COLLECT_QKV(model, tokenizer, prompt):
    activations = {}
    for each layer idx in model.layers:
        hook_q = register_hook(layer.q_proj, store in activations[idx]["Q"])
        hook_k = register_hook(layer.k_proj, store in activations[idx]["K"])
        hook_v = register_hook(layer.v_proj, store in activations[idx]["V"])
    
    model(tokenize(prompt))
    remove all hooks
    return activations
```

### 5.2 Head Scoring Phase

```
function COMPUTE_HEAD_SCORES(Q, K, V, num_q_heads, num_kv_heads, head_dim):
    T = sequence_length
    Q_heads = reshape(Q, T, num_q_heads, head_dim)
    K_heads = reshape(K, T, num_kv_heads, head_dim)
    V_heads = reshape(V, T, num_kv_heads, head_dim)
    
    attn_weighted_v = zeros(num_q_heads, num_kv_heads, head_dim)
    attn_flat = zeros(num_q_heads, num_kv_heads, T*T)
    
    for i in range(num_q_heads):
        for h in range(num_kv_heads):
            scores = Q_heads[:,i,:] @ K_heads[:,h,:]^T / sqrt(head_dim)
            attn = softmax(scores)
            attn_flat[i,h] = flatten(attn)
            attn_weighted_v[i,h] = mean(attn @ V_heads[:,h,:])
    
    kv_importance = sum(abs(attn_weighted_v), dims=(0,2))
    q_importance = sum(abs(attn_flat), dims=(1,2,3))
    
    kv_similarity = pairwise_cosine(attn_weighted_v, dim=0)
    q_similarity = pairwise_cosine(attn_flat, dim=0)
    
    return kv_importance, q_importance, kv_similarity, q_similarity
```

### 5.3 Head Merging Phase

```
function MERGE_HEADS(W, importance, similarity, src_heads, tgt_heads, head_dim):
    W_heads = reshape(W, src_heads, head_dim, d_in)
    active = list(range(src_heads))
    active_importance = importance.clone()
    
    while len(active) > tgt_heads:
        # Find most similar pair
        best_sim = -inf
        for i,j in combinations(active, 2):
            if similarity[i,j] > best_sim:
                best_sim = similarity[i,j]
                best_i, best_j = i, j
        
        # Weighted merge
        w = active_importance[best_i] / (active_importance[best_i] + active_importance[best_j])
        merged = w * W_heads[best_i] + (1-w) * W_heads[best_j]
        
        # Update
        W_heads[best_i] = merged
        active_importance[best_i] += active_importance[best_j]
        remove best_j from active
    
    return reshape(W_heads, tgt_heads * head_dim, d_in)
```

### 5.4 Full Projection

```
function PROJECT(W_src, target_shape, layer_idx, role):
    if role in {k_proj, v_proj}:
        head_imp, head_sim = head_scores[layer_idx].kv
        W_merged = MERGE_HEADS(W_src, head_imp, head_sim, src_kv=3, tgt_kv=2)
        W_proj = SVD_columns(W_merged, d_model=576 -> 512)
    
    elif role == q_proj:
        head_imp, head_sim = head_scores[layer_idx].q
        W_merged = MERGE_HEADS(W_src, head_imp, head_sim, src_heads=9, tgt_heads=8)
        W_proj = SVD_columns(W_merged, d_model=576 -> 512)
    
    elif role == o_proj:
        head_imp, head_sim = head_scores[layer_idx].q
        W_T = transpose(W_src)  # head dim becomes rows
        W_merged = MERGE_HEADS(W_T, head_imp, head_sim, src_heads=9, tgt_heads=8)
        W_proj = transpose(W_merged)
        W_proj = SVD_rows(W_proj, d_model=576 -> 512)
    
    else:
        W_proj = SVD_fallback(W_src, target_shape)
    
    return energy_preserve(W_proj, W_src)
```

---

## 6. Implementation Plan

### 6.1 New File: `paradom/core/activation_aware_projector.py`

**Status**: Implemented (see file)

Contains:
- `collect_qkv_activations()` - hook-based calibration data collection
- `ActivationAwareProjector` class with:
  - `calibrate()` - run calibration on source model
  - `project()` - main projection entry point
  - `_compute_head_scores()` - attention-based scoring
  - `_merge_heads()` - greedy weighted head merging
  - `_svd_fallback()` - standard SVD for non-attention weights
  - `get_head_report()` - debugging diagnostics

### 6.2 Modify: `paradom/mappings/transformer_to_transformer.py`

Changes needed:
1. Import `ActivationAwareProjector` and `collect_qkv_activations`
2. In `__init__`, add optional `projector` parameter
3. In `convert()`, for attention weights (k/v/q/o_proj), route through projector
4. Fall back to existing `_apply_swap` for non-attention weights

```python
# In TransformerToTransformerMapper.__init__:
self.projector = projector  # Optional[ActivationAwareProjector]

# In convert(), for attention weights:
if self.projector is not None and role in ATTENTION_ROLES:
    out = self.projector.project(wp.tensor, target_shape, src_i, role)
else:
    out = self._apply_swap(wp, target_shape, ...)
```

### 6.3 Modify: `scratch/hf_end_to_end_test.py`

Changes needed:
1. After loading source model, run calibration
2. Create projector with calibration data
3. Pass projector to mapper for Test B

```python
# After loading source model:
projector = ActivationAwareProjector(SOURCE_CONFIG, target_b)
projector.calibrate(model, tokenizer, PROMPT)

# In run_test:
mapper = TransformerToTransformerMapper(projector=projector, ...)
```

### 6.4 Modify: `paradom/core/swap_engine.py`

Optional: Add `collect_qkv_activations` as a wrapper around the new module, or update the existing `collect_kv_activations` to also collect Q.

---

## 7. Expected Challenges and Mitigations

### 7.1 Calibration Data Approximation

**Challenge**: Calibration data comes from the source model, but the target model will have different weights and thus different activations. The attention patterns will differ.

**Mitigation**: The RELATIVE importance of heads should be preserved because:
- Important heads carry the most information (this is a property of the weights)
- Redundant heads overlap with other heads (this is structural)
- These properties transfer across the projection

**Evidence**: The existing PCA approach failed because it optimized for variance, not attention quality. Our approach specifically targets attention patterns, which are more robust to weight changes.

### 7.2 Single Calibration Prompt

**Challenge**: One prompt may not represent the model's full behavior.

**Mitigation**: For the 3->2 kv_head reduction, the merging decision is based on structural similarity of attention patterns, which is relatively stable across prompts. A single short prompt (10-50 tokens) is sufficient.

**Future improvement**: Support multiple calibration prompts and average the scores.

### 7.3 Greedy vs Optimal Merging

**Challenge**: Greedy merging might not find the global optimum for larger reductions (e.g., 8->2).

**Mitigation**: For our case (3->2 kv_heads, 9->8 q_heads), greedy IS optimal:
- 3->2: only 3 possible pairs, greedy checks all
- 9->8: only removes one head, greedy finds the best

### 7.4 Energy Distribution

**Challenge**: Head merging changes the energy distribution. The merged head might have different magnitude than individual heads.

**Mitigation**: Energy preservation rescaling:
```python
scale = sqrt(src_energy / proj_energy)
```
This maintains the overall scale, preventing activation magnitude mismatches.

### 7.5 Compatibility with MagneticProjector

**Challenge**: The existing MagneticProjector computes shared spectral bases across layers. Our per-layer approach is different.

**Mitigation**: They can coexist:
- ActivationAwareProjector for attention weights (k/v/q/o_proj)
- MagneticProjector for non-attention weights (FFN, norms)
- Or replace MagneticProjector entirely for attention weights

---

## 8. Why This Should Work Better Than SVD

### 8.1 Preserves Head Structure

SVD creates linear combinations across ALL 192 rows. Our approach preserves clean 64-dim head blocks by merging entire heads, not individual rows.

```
SVD result:     row_i = a*head0 + b*head1 + c*head2  (mixed)
Ours result:    row_0:31 = merged(head_a, head_b)     (clean pair)
                row_32:63 = head_c                     (clean single)
```

### 8.2 Uses Functional Information

SVD uses only weight magnitudes. Our approach uses:
- Attention patterns (how heads are actually used)
- Value contributions (how much information flows through each head)
- Head redundancy (which heads can substitute for each other)

### 8.3 Optimizes for the Right Metric

SVD minimizes: `||W_src - W_tgt||_F` (Frobenius norm)
We maximize: `similarity(attention_patterns_src, attention_patterns_tgt)`

The second metric is what actually determines output quality.

### 8.4 Empirical Evidence

From CONCEPTS_LOG.md:
- SVD: CKA=0.9990, E_ratio=0.58, output=GARBAGE
- Head-aware (weight cosine): CKA=0.8635, output=GARBAGE
- PCA on activations: CKA=0.8174, output=WORSE

Our approach is different from all of these:
- Not SVD (preserves head structure)
- Not weight-cosine (uses attention patterns, not weight similarity)
- Not PCA (optimizes attention quality, not variance)

### 8.5 Computational Cost

- Calibration: 1 forward pass (~5s on CPU for 135M model)
- Head scoring: O(num_heads * num_kv * T^2 * d) per layer (~1ms per layer)
- Head merging: O(src_heads^2) per layer (~negligible)
- Total for 30 layers: ~5s calibration + ~30ms scoring = ~5s

This is well under the 5-minute budget.

---

## 9. Testing Plan

### 9.1 Unit Tests

1. Test head scoring with known attention patterns
2. Test merging preserves expected information
3. Test SVD fallback for non-attention weights
4. Test energy preservation scaling

### 9.2 Integration Tests

1. Run Test B from hf_end_to_end_test.py with projector
2. Compare output coherence with SVD baseline
3. Compare CKA scores (expect lower CKA but better output)
4. Measure total projection time

### 9.3 Diagnostic Checks

1. Print head reports for each layer (importance, similarity, merge pairs)
2. Verify head structure is preserved (each 64-dim block is clean)
3. Compare attention patterns between source and projected model
4. Check energy ratios for all projected weights

---

## 10. Files to Create/Modify

### Create:
- `paradom/core/activation_aware_projector.py` (DONE)

### Modify:
- `paradom/mappings/transformer_to_transformer.py` - integrate projector
- `scratch/hf_end_to_end_test.py` - use projector in Test B
- `paradom/core/swap_engine.py` - optional: update collect_kv_activations

### Reference:
- `scratch/CONCEPTS_LOG.md` - document results
- `scratch/debug_alternatives.py` - add projector as Strategy 6
