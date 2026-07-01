# Phase 1: Foundation & Prototype

**Document:** PARADOM-PHASE-001  
**Duration:** Week 1  
**Status:** NOT STARTED  
**Goal:** Prove the core hypothesis with a working minimal prototype.

---

## Phase 1 Objective

> Demonstrate that meaningful weight transfer is possible between two transformer variants by building a minimal but functional conversion pipeline and measuring the quality of the output.

---

## Success Criteria

- [ ] Can load LLaMA 3 8B weights from HuggingFace
- [ ] Can convert LLaMA 3 8B to Mistral 7B architecture shape (same-family test)
- [ ] Converted model achieves ≥70% performance retention on perplexity benchmark
- [ ] Pipeline runs end-to-end in <30 minutes on a single machine
- [ ] All core modules have unit tests with >80% coverage

---

## Week 1: Setup & Core Infrastructure

### Days 1–2: Project Bootstrap

**Tasks:**
- Set up Python package structure (`awfe/`)
- Configure CI/CD (GitHub Actions)
- Set up testing framework (pytest)
- Write base data structures: `ModelSnapshot`, `LayerGroup`, `ArchitectureSpec`
- Implement `ModelLoader` for SafeTensors and HF Hub

**Deliverable:** `awfe-core` package installable via `pip install -e .`

**Key Code — Model Loader:**
```python
class ModelLoader:
    def from_hub(self, model_id: str, cache_dir: str = None) -> ModelSnapshot:
        """Download and load model weights from HuggingFace Hub."""
        
    def from_local(self, path: str) -> ModelSnapshot:
        """Load weights from local checkpoint directory."""
        
    def from_safetensors(self, path: str) -> ModelSnapshot:
        """Load directly from .safetensors file(s)."""
```

**Risks:** HuggingFace API changes, checkpoint format variations  
**Mitigation:** Abstract behind adapter layer; test against 3+ model families

---

### Days 3–4: Architecture Parser

**Tasks:**
- Implement `ArchitectureParser.detect_architecture()`
- Build configuration extractors for: LLaMA 3, Mistral, Falcon, Gemma
- Implement `LayerGroupExtractor` — group raw weight tensors into semantic groups
- Write architecture YAML schema and validator
- Implement `ArchitectureSpec.from_yaml()`

**Test Matrix:**

| Model | Architecture Detected | Layer Groups Parsed |
|---|---|---|
| meta-llama/Llama-3-8B | ✅ | ✅ |
| mistralai/Mistral-7B-v0.3 | ✅ | ✅ |
| tiiuae/falcon-7b | ✅ | ✅ |
| google/gemma-7b | ✅ | ✅ |

**Deliverable:** `awfe inspect <model_id>` CLI command working

---

## Week 1: Mathematical Core

### Days 5–6: Decomposition Engine

**Tasks:**
- Implement `WeightDecomposer.svd_decompose()` with energy thresholding
- Implement `WeightDecomposer.pca_project()` for dimension changes
- Implement `WeightDecomposer.eigendecompose_attention()` — the critical mapping step
- Performance: SVD must complete for a 7B model layer in <5 seconds
- Write numerical stability tests (check for NaN, inf in decompositions)

**Critical Benchmark:**
```python
# This must pass:
W = torch.randn(4096, 4096)  # Typical attention matrix
U, S, Vh = decomposer.svd_decompose(W, energy_threshold=0.99)
W_reconstructed = U @ torch.diag(S) @ Vh
reconstruction_error = torch.norm(W - W_reconstructed) / torch.norm(W)
assert reconstruction_error < 0.01  # <1% reconstruction error
```

**Deliverable:** Decomposer module with full test suite

---

### Days 7: First Mapper — Same-Family Transformer

**Tasks:**
- Implement `TransformerToTransformerMapper`
- Handle: hidden dim changes, head count changes, GQA vs MHA, FFN ratio changes
- Implement `DimensionProjector` (PCA compress + pad expand)
- Implement normalization translation (LayerNorm ↔ RMSNorm weight rescaling)
- Integration test: LLaMA 3 8B → custom transformer config (different dimensions)

**Key Correctness Test:**
```python
# Same architecture, same dimensions = weights should be identical
result = engine.convert(source="llama-3-8b", target=llama_3_8b_spec)
for layer_name, weights in result.layers.items():
    original = source_model.layers[layer_name]
    torch.testing.assert_close(weights, original, atol=1e-5, rtol=1e-5)
```

**Deliverable:** Working same-family conversion, measurable quality

---

## Week 1: Validation & First Prototype

### Days 7: Validation Suite

**Tasks:**
- Implement `perplexity_delta()` using WikiText-2 test set
- Implement `output_cosine_similarity()` — compare logit distributions on 100 prompts
- Implement `ValidationReport` dataclass and JSON serialization
- Implement quality tier classification
- Add `awfe validate` CLI command

**Benchmark Protocol:**
```bash
# Standard validation run
awfe validate \
  --source meta-llama/Llama-3-8B \
  --converted ./output/converted_model \
  --test-tokens 256000 \
  --report ./reports/phase1_validation.json
```

---

### Days 7: End-to-End Integration & Documentation

**Tasks:**
- Wire all modules into `AWFE.convert()` main pipeline
- Implement `awfe convert` CLI command
- End-to-end test: HuggingFace download → conversion → validation → report
- Write developer setup documentation
- Write architecture decision records (ADRs) for all major design choices
- Prepare Phase 1 technical report

**Phase 1 Demo:**
```bash
# This must work by end of Month 3:
awfe convert \
  --source meta-llama/Llama-3-8B \
  --target-arch transformer \
  --target-config configs/custom_transformer_7b.yaml \
  --output ./output/phase1_demo \
  --validate

# Expected output:
# ✅ Model loaded: LLaMA 3 8B (8.03B params)
# ✅ Architecture parsed: 32 transformer layers
# ✅ Conversion complete: 12m 34s
# ✅ Validation: Perplexity ratio 1.08 (EXCELLENT)
# ✅ Output similarity: 0.91 (EXCELLENT)
# 📁 Saved to: ./output/phase1_demo
```

---

## Phase 1 Technical Decisions

### Decision 1: Start with Same-Family Conversion

**Rationale:** Same-family conversion (LLaMA → similar transformer) is mathematically trivial when dimensions match, but tests all pipeline infrastructure. It lets us validate the end-to-end pipeline and measure what "100% retention" looks like before we introduce the hard cross-architecture problem.

### Decision 2: SafeTensors as Primary Format

**Rationale:** SafeTensors is the dominant format for open-source LLMs on HuggingFace. It's fast, memory-safe, and supports lazy loading — critical for large models.

### Decision 3: Layer-by-Layer Processing

**Rationale:** Processing one layer at a time allows AWFE to handle models larger than RAM by streaming. Design the pipeline for streaming from the start.

### Decision 4: No Training Data in Phase 1

**Rationale:** If Phase 1 works without calibration data, it proves the mathematical mapping alone carries meaningful quality. Calibration is Phase 3.

---

## Phase 1 Resource Requirements

| Resource | Requirement | Notes |
|---|---|---|
| Developer(s) | 1 senior ML engineer | Full-time |
| Compute | 1× machine, 64GB RAM | For testing 7B models |
| GPU | Optional (1× RTX 3090) | Only for calibration tests |
| Storage | 500GB | Model caches |
| API Access | HuggingFace Pro | For gated models |

---

## Phase 1 Exit Criteria

Phase 1 is complete when ALL of the following are true:

1. ✅ `awfe convert` command runs successfully for same-family transformer conversion
2. ✅ Validation report shows ≥70% quality retention (Tier 2 or better)
3. ✅ Test coverage ≥80% on all core modules
4. ✅ Documentation complete for core modules
5. ✅ Phase 1 technical report written with honest assessment of results
6. ✅ Phase 2 plan updated based on Phase 1 learnings
