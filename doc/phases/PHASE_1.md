# Phase 1: Foundation & Spectral Prototype

**Document:** PARADOM-PHASE-001  
**Duration:** Week 1  
**Status:** IN PROGRESS  
**Goal:** Bootstrap the Streaming Swapper Engine and prove Spectral Mirroring.

---

## Phase 1 Objective

> Demonstrate **Functional Equivalence ($3=4-1$)** by redressing a source model into a target architecture using the **Streaming Swapper Engine**. Prove that intelligence can be mirrored disk-to-disk on consumer hardware (<16GB RAM).

---

## Success Criteria

- [ ] **Low-Resource Streaming**: Convert a 7B model using <16GB system RAM.
- [ ] **Spectral Integrity**: Achieve ≥0.99 spectral energy preservation in mirrored layers.
- [ ] **Mmap Snapshotting**: Zero-copy disk access for target weight writing.
- [ ] **Modular Swap**: Successful swap of at least 3 layers using `AttentionMirror`.
- [ ] **Code Quality**: 100% type-hinted, professional modular architecture.

---

## Week 1: Setup & Swapper Core

### Days 1–2: Paradom Bootstrap (Streaming First)

**Tasks:**
- Initialize `paradom/` package structure.
- Build the `MmapSnapshot` engine for lazy loading/writing.
- Implement `StreamingSwapper` orchestrator — processes model layer-by-layer.
- Write `ResourceMonitor` to verify the <16GB RAM constraint.

**Deliverable:** `paradom-core` with working disk-to-disk mmap pipeline.

---

### Days 3–4: Spectral Decomposer

**Tasks:**
- Implement `WeightDecomposer.eig_mirror()` for attention spectral extraction.
- Build `CanonicalBridge` — intermediate representation for weight derivations.
- Implement `Redresser` logic (the $3=4-1$ engine).
- numerical stability check for large-scale eigen-shunts.

**Mathematical Milestone:**
```python
# MIRROR TEST:
product = W_Q @ W_K.T
spectrum = decomposer.extract_energy(product)
target_weights = redresser.derive_from_spectrum(spectrum, target_config)
# Energy preservation must be > 99.9%
```

---

### Days 5–7: Modular Swapper & First Derivative

**Tasks:**
- Implement `AttentionSwapper` (Transformer → Transformer Mirror).
- Implement `GatedSwapper` (FFN Functional Equivalence).
- **Phase 1 Demo**: Convert Llama-3-8B to a custom "Sovereign derivative" with mirrored layers.
- Generate `SpectralValidationReport`.

**Phase 1 Demo Command:**
```bash
paradom swap \
  --source meta-llama/Llama-3-8B \
  --target-spec signatures/sovereign_7b.yaml \
  --mode streaming \
  --ram-limit 12GB
```

---

## Phase 1 Exit Criteria

1. ✅ `paradom swap` runs on <16GB RAM.
2. ✅ Spectral Energy Ratio ≥ 0.99.
3. ✅ Disk-to-disk conversion completes in <20 mins for 7B models.
4. ✅ Functional Equivalence proved via Layer-0 logit matching.
