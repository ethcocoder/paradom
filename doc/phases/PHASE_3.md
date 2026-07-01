# Phase 3: Day 3 — Intelligence Mapping & Hardening

**Document:** PARADOM-PHASE-003  
**Duration:** Day 3  
**Prerequisites:** Phase 1 (Discovery) and Phase 2 (LLM Handlers) complete.  
**Goal:** Implement the Equivalence Identifier (CKA) and refine the mathematical swap types to ensure high-quality intelligence transfer.

---

## Day 3 Objective

> Move from "blind swapping" to "intelligent mapping." Implement Centered Kernel Alignment (CKA) to verify that weights in Model A and Model B are actually doing the same job.

---

## Success Criteria

- [ ] `EquivalenceIdentifier` functional: Computes CKA scores between layers.
- [ ] `SwapEngine` supports all 4 swap types: Direct, Projected, Tucker, and OT.
- [ ] Threshold-based routing implemented: Automatically chooses the best swap type based on CKA.
- [ ] Mathematical stability verified for Cross-Architecture projection.
- [ ] Transformer → Mamba mapping logic achieves >0.50 mean CKA on sample layers.

---

## Schedule & Tasks

### Morning: The Rosetta Stone (08:00 - 12:00)
*   **Task 1**: Implement `CKASimilarityMatcher`. Supports linear CKA for fast activation comparison.
*   **Task 2**: Build the `EquivalenceMap` generator — links source weights to target slots with confidence scores.
*   **Task 3**: Create `paradom identify` CLI command to visualize mappings before swapping.

### Afternoon: Mathematical Swapping (13:00 - 17:00)
*   **Task 4**: Finalize `ProjectedSwap` using SVD + Procrustes rotation for dimensional alignment.
*   **Task 5**: Implement `TuckerSwap` for vision paradigms (CNN filters → ViT patches).
*   **Task 6**: Implement `OptimalTransportSwap` (OT) as a fallback for low-CKA cross-paradigm translations.

### Evening: Hardening & Validation (18:00 - 20:00)
*   **Task 7**: Write unit tests for all 4 swap types using synthetic tensors.
*   **Task 8**: Conduct "Drift Test": Measure activation error after projected swap vs direct copy.

---

## Technical Milestone: The Intelligence Match

```python
# Day 3 Demo: Intelligence Mapping
matcher = EquivalenceIdentifier(method="cka")

# Compare Layer 0 of Llama and Mamba
score = matcher.evaluate(layer_llama, layer_mamba)
print(f"Functional Equivalence Score: {score.cka:.3f}")

if score.cka > 0.70:
    engine.execute_swap(type="direct")
elif score.cka > 0.40:
    engine.execute_swap(type="projected")
else:
    engine.execute_swap(type="ot")
```

---

## Day 3 Exit Criteria
1. ✅ `paradom identify` command produces a detailed JSON mapping report.
2. ✅ All 4 swap types verified with mathematical precision.
3. ✅ No regressions in streaming RAM usage (< 2GB).
