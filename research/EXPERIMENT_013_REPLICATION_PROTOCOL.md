# Experiment 013: Independent-Seed Replication on an Untouched Test Set

## Purpose

Experiment 012 produced a promising two-layer Mamba hybrid. Experiment 013 tests whether that outcome is stable across random initialization and an English text set that is never used for fitting, gate scheduling, or intermediate model selection.

## Design

Three complete runs use independent seeds: `20260821`, `20260822`, and `20260823`. Every run retains Experiment 012’s fixed procedure: 256 WikiText-2 training sequences for calibration, the same layer-0 and layer-1 update budgets, state sizes 64 and 96, teacher-frozen functional targets, sequential gates, and equal direct versus random Mamba budgets.

| Dataset role | Corpus partition | Permitted use |
|---|---|---|
| Calibration | WikiText-2 train | Local Mamba fitting and gated distillation only |
| Development observation | WikiText-2 validation | Intermediate gate-curve measurement only; no gradient updates |
| Final test | WikiText-2 test | Final loss, perplexity, and deterministic generations only |

## Primary Outcomes

The principal outcome is final causal-language-model loss on the untouched test sequences after both attention layers have gate value `α = 1`. Secondary outcomes are the teacher gap, local layer-1 fit, direct-versus-random paired loss difference, and fixed-prompt English continuations.

A stable result requires every seed to preserve the exact source endpoint at `α = 0`, to produce finite loss after two replacements, and to retain grammatical English on at least the scientific-research and explorer prompts. The random versus direct comparison is descriptive with three seeds; it is not treated as a reliable significance claim.
