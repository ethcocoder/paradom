# Contributing to AWFE

Welcome to AWFE! We're building the world's first general-purpose cross-architecture neural network weight converter. Every contribution matters.

---

## How to Contribute

### 1. Add a New Architecture Definition

Create a YAML file in `configs/architectures/`:

```yaml
# configs/architectures/my_arch.yaml
name: "MyArch-7B"
type: "transformer"
model:
  vocab_size: 32000
  hidden_size: 4096
  ...
```

Then add a parser in `awfe/architectures/my_arch.py`.

### 2. Add a New Mapping Strategy

Subclass `BaseMapper` in `awfe/mappings/`:

```python
class MySourceToMyTargetMapper(BaseMapper):
    def map_attention(self, source): ...
    def map_ffn(self, source): ...
    def map_norm(self, source): ...
    def map_embedding(self, source): ...
    def map_head(self, source): ...
```

Register in `awfe/mappings/registry.py`.

### 3. Report Conversion Quality

If you've run AWFE and have benchmark results, share them! Open a PR adding your results to `BENCHMARK_RESULTS.md`.

---

## Development Setup

```bash
git clone https://github.com/your-org/awfe
cd awfe
pip install -e ".[dev]"
pytest tests/
```

---

## Code Standards

- Type annotations on all public functions
- Docstrings on all classes and methods
- Unit tests required for new modules (>80% coverage)
- No hard-coded model paths — always configurable

---

# Known Limitations

**Document:** AWFE-LIMITS-001  
*(We believe in radical honesty about what AWFE can and cannot do.)*

---

## What AWFE Cannot Do

### 1. It is NOT Replication
AWFE cannot replicate the full capability of a source model. It transfers *most* of the intelligence, but some is always lost. Think of it like translating a book — the meaning transfers, but some nuance is lost.

### 2. It Cannot Transfer Architecture-Specific Strengths
If a model was specifically trained to exploit properties of Transformer attention (e.g., associative recall over long context), those properties may be weak in a converted Mamba model because SSMs have different inductive biases.

### 3. It Does NOT Replace Fine-Tuning
For production sovereign AI deployment, AWFE should be seen as the **starting point**, not the endpoint. After conversion, fine-tuning on local language data will significantly improve quality for specific use cases.

### 4. Cross-Architecture Quality Varies
Same-architecture conversion is reliable (85–95%). Cross-architecture conversion (Transformer → Mamba) is less predictable (50–75%) and depends heavily on the source model and calibration data available.

### 5. It Cannot Handle Proprietary Models
AWFE only works with models where weights are openly available. GPT-4, Claude, Gemini, and other proprietary models cannot be converted.

### 6. Very Large Models (>70B) Are Experimental
AWFE supports 70B+ models in streaming mode, but this is less tested than 7B–13B conversions. Expect longer times and potentially lower quality at extreme scale.

---

## Current Research Uncertainties

- The exact mechanism by which Transformer attention eigenstructure maps to SSM state dynamics is still being validated empirically.
- Quality predictions before running conversion are approximate.
- The impact of the source model's training data on conversion quality is not fully understood.

---

## Honest Assessment of Probability of Success

| Goal | Confidence | Notes |
|---|---|---|
| Same-arch conversion ≥80% quality | High | Mathematically well-founded |
| Transformer → Transformer cross-config | High | Proven techniques |
| Transformer → Mamba ≥65% (with calibration) | Medium | Novel; 2 known partial results |
| Zero-shot cross-arch ≥65% | Low-Medium | Highly ambitious |
| Full sovereign AI deployment | Medium | Depends on Phase 1–3 results |

We will update this document as we learn more. If Phase 2 shows that Transformer → Mamba conversion quality is consistently below 50%, we will update the roadmap and communicate this openly.

**The founding team believes the core hypothesis is sound. But we also know it is novel. We build in public, and we report our results honestly — including failures.**
