# Phase 1: Proof of Concept — Number Equivalence

**Document:** PARADOM-PHASE-001
**Duration:** Months 1–3
**Goal:** Prove the core hypothesis on small models. Show that swapping weight "products" between architectures preserves meaningful intelligence.

---

## Phase 1 Objective

> Before building the full framework, we must answer ONE question empirically:
> "If I take weight W from Model A and place it at the functionally equivalent position in Model B — does the intelligence survive?"

This phase builds the minimum viable experiment to answer that question with real numbers.

---

## The Core Experiment

```
Setup:
  Source: 2-layer Transformer (tiny, ~10M params — fast to work with)
  Target: 2-layer Mamba SSM (equivalent depth and width)
  
  Both trained on the same small dataset (WikiText-103 subset)
  Both evaluated on the same test set

Experiment:
  Step 1: Train Transformer from scratch → perplexity = X
  Step 2: Train Mamba from scratch → perplexity = Y
  Step 3: Swap top 20% weights from Transformer → Mamba → perplexity = Z
  
  Key question: Is Z meaningfully closer to X than random initialization?
  
  If Z < Y + 20%:  Hypothesis SUPPORTED → build full framework
  If Z ≈ Y:        Hypothesis UNCERTAIN → investigate why, refine
  If Z > Y + 50%:  Hypothesis CHALLENGED → revise approach
```

This experiment is small enough to run on a laptop in under an hour. It is the most important thing Phase 1 does.

---

## Success Criteria

- [ ] Core experiment runs end-to-end on a laptop
- [ ] Weight equivalence is demonstrable on toy models (2-layer Transformer ↔ Mamba)
- [ ] `paradom identify` command works — shows equivalence map between two models
- [ ] Basic swap engine handles direct and projected swaps
- [ ] Validation suite produces a SwapValidationReport
- [ ] Pipeline runs in streaming mode (one layer at a time, <4GB RAM)
- [ ] Unit test coverage ≥ 80% on core modules
- [ ] Phase 1 findings documented honestly (success OR failure)

---

## Month 1: Infrastructure & Core Experiment

### Week 1–2: Project Bootstrap

**Tasks:**
- Set up Python package: `pip install -e .` works
- CI/CD pipeline (GitHub Actions): tests run on every commit
- Core data structures:

```python
# These must exist by end of Week 2
@dataclass
class WeightProduct:
    """A single weight tensor with its metadata."""
    name: str                    # e.g. "layers.0.self_attn.q_proj.weight"
    tensor: Tensor               # The actual numbers
    shape: tuple
    functional_role: FunctionalRole
    paradigm: str
    importance_score: float      # Set after ImportanceScorer runs
    
@dataclass  
class EquivalenceMap:
    """Maps source weights to target weight positions."""
    pairs: List[Tuple[WeightProduct, str]]  # (source_weight, target_layer_name)
    unmapped_source: List[str]              # Source weights with no equivalent
    uninitialized_target: List[str]         # Target weights with no source
    mean_cka: float
```

- Implement `ModelLoader` for SafeTensors and PyTorch checkpoints
- Implement `ArchitectureParser` for: LLaMA, Mistral, Mamba (these 3 first)

**Deliverable:** Package installs, loads a LLaMA 3 8B checkpoint, prints layer names

---

### Week 3–4: The Core Experiment

**Tasks:**
- Train tiny 2-layer Transformer (10M params) on WikiText-103 for 1 hour
- Train tiny 2-layer Mamba (10M params) on WikiText-103 for 1 hour
- Implement `FunctionalRoleMatcher` for Transformer ↔ Mamba
- Implement `ImportanceScorer.svd_spectrum()`
- Implement `SwapEngine.direct_swap()` and `SwapEngine.projected_swap()`
- Run the core experiment. Record results.

**Core Experiment Code:**
```python
from paradom import Paradom, TargetSpec

# Load both models
transformer = load_model("./tiny_transformer_trained.pt")
mamba_config = TargetSpec.from_yaml("configs/tiny_mamba.yaml")

# Run Paradom swap
engine = Paradom()
result = engine.swap(
    source=transformer,
    target=mamba_config,
    config=SwapConfig(
        importance_method="svd_spectrum",
        swap_fraction=0.20,
        streaming=False  # Small model, fits in RAM
    ),
    output_path="./output/transformer_as_mamba"
)

# Evaluate
mamba_from_scratch_ppl = evaluate_perplexity("./tiny_mamba_trained.pt")
mamba_from_swap_ppl    = evaluate_perplexity("./output/transformer_as_mamba")
transformer_ppl        = evaluate_perplexity("./tiny_transformer_trained.pt")

print(f"Transformer baseline:     {transformer_ppl:.2f}")
print(f"Mamba from scratch:       {mamba_from_scratch_ppl:.2f}")
print(f"Mamba from Paradom swap:  {mamba_from_swap_ppl:.2f}")
print(f"Quality preservation:     {mamba_from_scratch_ppl/mamba_from_swap_ppl:.1%}")
```

**Document results in:** `research/EXPERIMENT_001_RESULTS.md`

---

## Month 2: Equivalence Identifier & Streaming

### Week 5–6: CKA-Based Equivalence Identifier

**Tasks:**
- Implement `CKASimilarityMatcher.compute_cka()`
- Implement `EquivalenceIdentifier` — combines functional role matching + CKA
- Implement `paradom identify` CLI command:

```bash
# This must work by end of Week 6
paradom identify \
  --source meta-llama/Llama-3-8B \
  --target-config configs/mamba_7b.yaml \
  --output ./reports/equivalence_map.json

# Output example:
# Equivalence Map: LLaMA 3 8B → Mamba 7B
# ─────────────────────────────────────────
# Layer pairs identified: 96 / 128 source layers
# Unmapped source layers: 32 (will be ignored)
# Uninitialized target layers: 12 (will use Xavier init)
#
# Swap type distribution:
#   Direct swap:      45 layers (CKA avg: 0.84)
#   Projected swap:   38 layers (CKA avg: 0.61)
#   Tensor swap:       5 layers (CKA avg: 0.43)
#   OT swap:           8 layers (CKA avg: 0.27)
#
# Estimated quality tier: GOOD (70-85% retention)
# Estimated swap time: ~18 minutes
```

**Tests:**
```python
def test_cka_identical_models():
    """CKA between identical models should be 1.0"""
    model_a = load_tiny_transformer()
    model_b = copy.deepcopy(model_a)
    cka = compute_cka(model_a, model_b, layer="layers.0.self_attn")
    assert cka > 0.99

def test_cka_random_models():
    """CKA between random-init models should be near 0"""
    model_a = TinyTransformer()  # Random init
    model_b = TinyTransformer()  # Different random init
    cka = compute_cka(model_a, model_b, layer="layers.0.self_attn")
    assert cka < 0.20
```

---

### Week 7–8: Streaming Mode

**Tasks:**
- Implement `StreamingSwapper` — processes one layer at a time
- RAM monitoring: confirm peak stays under configured limit
- Test on a real 7B model (LLaMA 3 8B or Mistral 7B)
- Benchmark: measure actual RAM usage and time

**Streaming Correctness Test:**
```python
def test_streaming_matches_full():
    """Streaming mode must produce identical results to full-model mode."""
    result_full     = engine.swap(source, target, streaming=False)
    result_streaming = engine.swap(source, target, streaming=True)
    
    for layer_name in result_full.layers:
        torch.testing.assert_close(
            result_full.layers[layer_name],
            result_streaming.layers[layer_name],
            atol=1e-6, rtol=1e-6
        )
```

**RAM Verification:**
```python
def test_streaming_ram_usage():
    """Streaming mode must stay within configured RAM limit."""
    config = SwapConfig(streaming=True, max_ram_gb=4.0)
    
    with track_peak_ram() as tracker:
        engine.swap("meta-llama/Llama-3-8B", target, config)
    
    assert tracker.peak_gb < 4.0, \
        f"Peak RAM {tracker.peak_gb:.1f}GB exceeded 4.0GB limit"
```

---

## Month 3: Validation Suite & Phase 1 Report

### Week 9–10: Validation Suite

**Tasks:**
- Implement `SwapValidator.compute_cka_per_layer()`
- Implement LLM-specific: `perplexity_delta()` using WikiText-2
- Implement `quality_tier_classifier()`
- Implement `SwapValidationReport` with JSON export
- Implement `paradom validate` CLI command

```bash
paradom validate \
  --source meta-llama/Llama-3-8B \
  --swapped ./output/llama_as_mamba \
  --paradigm llm \
  --benchmark perplexity \
  --report ./reports/phase1_validation.json
```

---

### Week 11–12: Integration, CLI & Phase 1 Report

**Tasks:**
- Wire all modules: `paradom swap` end-to-end CLI works
- Integration test: HuggingFace download → identify → swap → validate → report
- Write `EXPERIMENT_001_RESULTS.md` — honest findings from core experiment
- Write Phase 1 Technical Report:
  - What worked
  - What didn't
  - What quality was actually achieved
  - Revised quality estimates for Phase 2
- Update Phase 2 plan based on Phase 1 learnings

---

## Phase 1 Exit Criteria

Phase 1 is complete when ALL of the following are true:

1. ✅ Core experiment complete with documented results
2. ✅ `paradom identify` works for Transformer → Mamba
3. ✅ `paradom swap` works end-to-end (same-arch, LLM paradigm)
4. ✅ Streaming mode verified: 7B model runs in <4GB RAM
5. ✅ Validation report generated with real quality numbers
6. ✅ Test coverage ≥ 80%
7. ✅ Phase 1 Technical Report written — honest, with real numbers

---

## Resource Requirements

| Resource | Requirement |
|---|---|
| Developer | 1 ML engineer (full-time) |
| Machine | 16GB RAM laptop or desktop |
| GPU | NOT required for Phase 1 |
| Storage | 100GB (model caches) |
| HuggingFace | Free tier sufficient |
| Time | 3 months |