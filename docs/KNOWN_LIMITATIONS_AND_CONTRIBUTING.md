# Known Limitations of Paradom

**Document:** PARADOM-LIMITS-001
**Version:** 2.0.0

*We believe in radical honesty. This document is written for researchers and engineers who need to know exactly what Paradom can and cannot do before relying on it.*

---

## What Paradom Is NOT

### 1. It Is Not Replication

Paradom transfers *most* of the intelligence, not all of it. A weight swap is an approximation — not an exact reproduction.

The analogy: translating a book from English to Amharic preserves the meaning, but some nuance is always lost. The better the translation method, the less is lost. But loss is always present.

**Practical implication:** For critical applications, always validate the converted model thoroughly before deployment. The `paradom validate` command is not optional — it is mandatory.

---

### 2. It Is Not a Replacement for Fine-Tuning in Production

For sovereign AI deployment in a specific language or domain, Paradom gives you the starting point. Fine-tuning on local data will dramatically improve quality for your specific use case.

```
Without fine-tuning:  Paradom swap gives you 65–82% of source quality
With fine-tuning:     Paradom swap + LoRA gives you 85–95% of source quality
                      (and a model specifically optimized for your language/domain)
```

Do not skip fine-tuning for production sovereign deployments.

---

### 3. It Cannot Handle Proprietary Models

Paradom only works with models whose weights are openly accessible. GPT-4, Claude, Gemini, Grok, and any other proprietary model cannot be used as source models.

**Safe source models (confirmed Apache 2.0 / MIT):**
- LLaMA 3 family (Meta)
- Mistral family (Mistral AI)
- Falcon family (TII)
- Gemma (Google — check specific version license)
- Qwen 2 (Alibaba)
- Phi-3 (Microsoft)

**Always verify the license** of any model before using it as a Paradom source.

---

### 4. Cross-Paradigm Swaps Are Weaker Than Same-Paradigm Swaps

The further apart the source and target paradigms are, the more quality is lost:

| Swap Type | Quality Retention |
|---|---|
| Same architecture (e.g., LLaMA 7B → LLaMA-like 7B) | 85–95% |
| Same paradigm, different arch (e.g., LLaMA → Mamba) | 65–82% |
| Related paradigm (e.g., Diffusion → Flow Matching) | 60–75% |
| Different paradigm (e.g., CNN → ViT) | 60–72% |
| Very different paradigm (e.g., DQN → PPO) | 55–68% |
| Highly different paradigm (e.g., GNN → Transformer) | 50–65% |

---

### 5. The A Matrix Cannot Be Swapped — It Must Be Derived

In the Transformer → Mamba conversion, the Mamba SSM's `A_log` parameter (which controls state decay dynamics) has NO direct Transformer equivalent. It is mathematically derived from the attention pattern matrix's eigenstructure.

This derivation is an approximation. It is the single greatest source of quality loss in LLM cross-architecture conversion.

**What this means in practice:** Transformer → Mamba conversion will always have a quality ceiling lower than Transformer → Transformer conversion, because one critical parameter must be derived rather than swapped.

---

### 6. Very Small Models May Lose More Quality

The Platonic Representation Hypothesis — our scientific foundation — is observed primarily in large models. Very small models (< 1B parameters) may not have converged representations, making the equivalence assumption weaker.

**Recommendation:** For models under 1B parameters, run the core experiment first (see Phase 1) and measure actual quality before committing to Paradom for that model size.

---

## Current Research Uncertainties

These are things we do not yet know and are actively investigating:

| Uncertainty | Status | Expected Resolution |
|---|---|---|
| Optimal swap fraction by paradigm | Phase 1 experiment will measure | Month 2–3 |
| State dimension sensitivity for Transformer → Mamba | Phase 2 study | Month 4 |
| Whether quality scales with source model size | Phase 2–3 | Month 6–9 |
| Whether GNN → Transformer quality is acceptable for production | Phase 3 | Month 9 |
| Long-term quality degradation on fine-tuned swapped models | Ongoing | Year 1+ |

---

## What To Do When Quality Is Too Low

If `paradom validate` returns a DEGRADED quality tier:

**Step 1: Try a higher swap fraction**
```bash
paradom swap ... --swap-fraction 0.35  # Up from default 0.20
```

**Step 2: Try a different importance method**
```bash
paradom swap ... --importance-method gradient_sensitivity
```

**Step 3: Run the calibration pass**
```bash
paradom calibrate \
  --converted ./output/my_model \
  --method activation_matching \
  --output ./output/my_model_calibrated
```

**Step 4: Fine-tune**

If calibration still leaves quality below acceptable, use standard LoRA fine-tuning:
```bash
# Use any standard fine-tuning framework (transformers, axolotl, unsloth)
# Even 10K examples of your target use case typically recovers 10–20% quality
```

**Step 5: Report the failure case**

Open a GitHub issue with your source model, target architecture, and validation report. Every failure case helps improve Paradom's mapping strategies.

---

## Probability of Success Estimates (Honest)

| Goal | Our Confidence | Notes |
|---|---|---|
| Same-arch swap ≥80% quality | High | Mathematically straightforward |
| Transformer → Mamba ≥65% quality | Medium | Novel; core experiment needed |
| Vision CNN → ViT ≥65% quality | Medium | Some published evidence supports this |
| RL DQN → PPO ≥60% quality (state encoder) | Medium | Unpublished territory |
| Diffusion → Flow ≥65% quality | Medium-High | Strong mathematical basis |
| All paradigms in one framework | High | Engineering challenge, not science risk |
| Sovereign AI deployment | Medium-High | Depends on Phase 1-3 results |

We will update these estimates as real experimental data comes in.

---

---

# Contributing to Paradom

**Document:** PARADOM-CONTRIB-001

---

## How to Contribute

### Contribution Type 1: New Architecture Definition

The simplest and highest-value contribution. Define a target architecture in YAML and validate that Paradom can convert to it.

```yaml
# configs/community/your_arch.yaml
name: "YourArch-7B"
type: "transformer"
paradigm: "llm"
contributed_by: "your-github-handle"
validated_source: "meta-llama/Llama-3-8B"
validated_quality: 0.78   # CKA score from your validation run

model:
  vocab_size: 32000
  hidden_size: 4096
  ...
```

**PR requirement:** Include your `swap_report.json` output as evidence of validation.

---

### Contribution Type 2: New Paradigm Mapping

Implement a new `BaseMapper` subclass for a conversion path Paradom doesn't yet support.

```python
# paradom/mappings/your_paradigm/source_to_target.py

class YourSourceToYourTargetMapper(BaseMapper):
    """
    Maps [source architecture] to [target architecture].

    Scientific basis: [cite the paper or explain the mathematical equivalence]
    Expected quality: [your estimate, with reasoning]
    Tested on: [which specific models you tested]
    """

    def map_attention(self, source: AttentionGroup) -> ...:
        ...

    def map_ffn(self, source: FFNGroup) -> ...:
        ...

    # etc.
```

**PR requirement:**
- Unit tests with ≥80% coverage
- Benchmark results on at least one model pair
- Honest quality estimate in docstring

---

### Contribution Type 3: Benchmark Results

Run Paradom on a model pair and submit your results to the public benchmark table.

```json
// benchmarks/community/your_result.json
{
  "contributor": "your-github-handle",
  "date": "2026-07-01",
  "source_model": "mistralai/Mistral-7B-v0.3",
  "target_arch": "mamba",
  "target_config": "configs/mamba_7b.yaml",
  "swap_fraction": 0.20,
  "importance_method": "svd_spectrum",
  "hardware": "16GB RAM, AMD Ryzen 7, no GPU",
  "conversion_time_minutes": 18,
  "peak_ram_gb": 3.2,
  "perplexity_source": 6.24,
  "perplexity_converted": 9.87,
  "quality_retention": 0.63,
  "quality_tier": "acceptable",
  "notes": "Quality improved to 0.71 after activation_matching calibration"
}
```

---

### Contribution Type 4: Sovereign AI Case Study

The most impactful contribution. Document a real deployment of Paradom for sovereign AI.

If you have used Paradom to build or prototype a model for a national, regional, or community AI initiative:
- Open an issue titled "Sovereign AI Case Study: [Your Organization/Region]"
- We will work with you to write up the case study
- You will be featured on the project website and invited to co-author related publications

---

## Development Setup

```bash
git clone https://github.com/your-org/paradom
cd paradom
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run type checking
mypy paradom/

# Run linting
ruff check paradom/

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

---

## Code Standards

- **Type annotations** on all public functions and class methods
- **Docstrings** on all classes: include scientific basis, expected quality, tested models
- **Unit tests** required for all new modules (≥80% coverage)
- **No hardcoded model paths** — always configurable via parameters
- **Honest docstrings** — if a mapper's quality is uncertain, say so
- **Log resource usage** — always log peak RAM and time for any benchmark

---

## Contribution Tiers

| Tier | What You Did | Recognition |
|---|---|---|
| **Bronze** | Submitted a validated architecture YAML | Listed in README contributors |
| **Silver** | Submitted a new mapper with benchmark results | Core contributor credit + Architecture Zoo listing |
| **Gold** | Documented a real-world sovereign AI deployment | Featured case study + invited paper co-authorship |
| **Platinum** | Contributed a full new paradigm (all mappers + benchmarks) | Acknowledged as paradigm maintainer |

---

## Community Channels

- **GitHub Discussions:** Technical questions, architecture design discussions
- **GitHub Issues:** Bug reports, feature requests
- **Discord:** (Link after launch) — real-time discussion
- **Monthly research call:** Open video call for contributors — first Monday of each month

---

## License

All contributions are made under Apache 2.0. By submitting a PR, you agree that your contribution is licensed under Apache 2.0 and you have the rights to make that contribution.
