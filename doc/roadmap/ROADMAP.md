# Master Roadmap: Paradom Framework

**Document:** PARADOM-ROADMAP-001  
**Version:** 1.0.0  
**Date:** 2026-06-30

---

## Executive Summary

Paradom will be developed over 5 weeks in 5 phases, moving from a minimal research prototype to a production-grade open-source framework that enables sovereign AI deployment worldwide.

---

## Full Timeline

```
2026
─────────────────────────────────────────────────────────────────────
Week:   1       2       3       4       5

PHASE 1: Foundation & Prototype
        ████████
        |      |
      START  Phase 1
             Complete

PHASE 2: Core Engine (Cross-Architecture)
                ████████
                |      |
              Start  Phase 2
                     Complete

PHASE 3: Calibration & Quality
                        ████████
                        |      |
                      Start  Phase 3
                             Complete

PHASE 4: Scale & Production
                                ████████
                                |      |
                              Start  Phase 4
                                     Complete

PHASE 5: Open Source Launch & Sovereignty
                                        ████████▶
                                        |
                                      PUBLIC
                                      LAUNCH
```

---

## Milestone Table

| Milestone | Phase | Target Week | KPI |
|---|---|---|---|
| First model loaded | 1 | Week 1 | ModelLoader works |
| Architecture auto-detected | 1 | Week 1 | 4 architectures parsed |
| Same-arch conversion | 1 | Week 1 | ≥80% quality |
| Full pipeline v0.1 | 1 | Week 1 | `paradom convert` CLI works |
| **Phase 1 Validation** | 1 | Week 1 | ≥70% retention, end-to-end |
| Transformer→Mamba mapping | 2 | Week 2 | Implementation complete |
| First cross-arch benchmark | 2 | Week 2 | ≥50% retention |
| 4+ conversion paths | 2 | Week 2 | Registry with 4 mappers |
| **Phase 2 Validation** | 2 | Week 2 | LLaMA→Mamba ≥50% |
| Zero-shot calibration | 3 | Week 3 | +15% quality gain |
| Full benchmark suite | 3 | Week 3 | HellaSwag, ARC, MMLU |
| **Phase 3 Validation** | 3 | Week 3 | Transformer→Mamba ≥65% |
| 70B model support | 4 | Week 4 | Streaming + parallel |
| REST API | 4 | Week 4 | API server running |
| PyPI release v1.0 | 4 | Week 4 | `pip install paradom` |
| **Phase 4 Validation** | 4 | Week 4 | First external users |
| Public GitHub launch | 5 | Week 5 | Open source |
| 500 GitHub stars | 5 | Week 6 | Community traction |
| First sovereign deployment | 5 | Week 7 | Real-world impact |
| Academic paper submitted | 5 | Week 8 | NeurIPS / ICLR |

---

## Technology Stack Decision

| Component | Decision | Rationale |
|---|---|---|
| Language | Python 3.11 | Industry standard for ML |
| Tensor Library | PyTorch 2.3+ | Most ML frameworks are PyTorch-native |
| Weight Format | SafeTensors | Fast, safe, universal for HF models |
| Math | SciPy + PyTorch linalg | Best SVD/eigendecomp support |
| CLI | Typer + Rich | Modern, pretty, type-safe |
| API | FastAPI | Async, fast, OpenAPI docs auto-generated |
| Testing | pytest | Standard |
| Docs | MkDocs + Material | Beautiful, markdown-native |
| CI/CD | GitHub Actions | Free, widely used |
| Package | PyPI | Standard distribution |
| Container | Docker | Reproducible environments |

---

## Resource Plan

### Team Requirements

| Phase | Core Team | Optional |
|---|---|---|
| Phase 1–2 | 1 ML engineer (senior) | 1 research advisor |
| Phase 3 | 1 ML engineer + 1 research engineer | HPC access |
| Phase 4 | 2 ML engineers + 1 backend engineer | Design for web UI |
| Phase 5 | Core team + community contributors | DevRel |

### Compute Requirements

| Phase | Minimum | Recommended |
|---|---|---|
| Phase 1 | 1× server, 64GB RAM | 1× server, 128GB RAM |
| Phase 2 | Same + 1× A100 for calibration | 2× A100 |
| Phase 3 | 2–4× A100 for benchmarking | 8× A100 |
| Phase 4 | HPC access for 70B testing | Cloud burst (GCP/AWS) |
| Phase 5 | Hosting for API service | Scalable cloud |

### Funding Path

| Option | Funding Amount | Likelihood |
|---|---|---|
| Self-funded (bootstrapped) | $0 | High — start here |
| Academic grant (EU Horizon, NSF) | $50K–$200K | Medium |
| African AI Development Fund | $20K–$100K | Medium |
| Open-source startup grant (GitHub, HF) | $10K–$50K | High |
| Seed investment (if commercialized) | $500K–$2M | Long-term |

---

## Risk Register

| Risk | Probability | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Core hypothesis fails (weights can't transfer cross-arch) | Medium | Critical | Start with same-arch; iterate; publish null results | Research lead |
| Key person dependency | High | High | Document everything; open-source early | Project lead |
| HuggingFace API changes break loader | Medium | Medium | Abstract loader; version-pin; test against multiple HF versions | ML engineer |
| SVD too slow for 70B | High | Medium | Randomized SVD; blocked SVD; parallelism | ML engineer |
| Legal issues with model weights | Low | High | Use only Apache 2.0 / permissive licensed source models | Project lead |
| Compute costs exceed budget | Medium | Medium | Start small (7B); use free tiers; seek grants | Project lead |
| Academic competition publishes similar work first | Medium | Low | Publish blog posts early to establish priority | Research lead |

---

## KPI Dashboard (Check Weekly)

### Technical KPIs

| KPI | Phase 1 Target | Phase 3 Target | Phase 5 Target |
|---|---|---|---|
| Conversion quality (same-arch) | ≥70% | ≥85% | ≥90% |
| Conversion quality (cross-arch) | N/A | ≥65% | ≥75% |
| Largest model supported | 7B | 13B | 70B |
| Supported source architectures | 2 | 4 | 8+ |
| Supported target architectures | 2 | 4 | 8+ |
| Conversion time (7B, same-arch) | <30 min | <15 min | <10 min |
| Test coverage | ≥80% | ≥85% | ≥90% |

### Community KPIs (Phase 5+)

| KPI | 3 Weeks Post-Launch | 6 Weeks | 12 Weeks |
|---|---|---|---|
| GitHub stars | 500 | 1,500 | 5,000 |
| PyPI downloads / month | 1,000 | 5,000 | 20,000 |
| Community contributors | 5 | 15 | 40 |
| Supported architectures (community) | +2 | +5 | +10 |
| Sovereign deployments | 1 | 3 | 10 |

---

## Communication Plan

### Internal (During Development)

- Daily: progress log in `CHANGELOG.md`
- Weekly: phase review document
- Per-milestone: benchmark report

### External (From Phase 4+)

- Week 4: Blog post — "Why We're Building Paradom"
- Week 5: Launch post — "Paradom v1.0: Convert Any Model to Any Architecture"
- Week 6: Technical deep-dive — "The Mathematics of Weight Transfer"
- Week 7: Sovereignty case study — "How Paradom Powers Ethiopia's First Sovereign AI"
- Week 8: Academic paper submission

---

## Definition of Project Success

**Minimum Viable Success (Phase 3):**
> Paradom demonstrably transfers meaningful intelligence from transformer models to at least one different architecture, with ≥65% quality retention and no training data required.

**Full Success (Phase 5):**
> Paradom is a widely-used open-source framework. At least one sovereign AI system has been built using Paradom. An academic paper is under review. The framework has a vibrant contributor community.

**Transformative Success (Long-term):**
> Cross-architecture weight conversion becomes standard practice, the way transfer learning is today. Paradom is cited as a foundational tool. Nations that previously had no path to sovereign AI have used Paradom to build their own models.
