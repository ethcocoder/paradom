# Experiment 015: Adaptive Interface Matching for Two-Layer Transformer-to-Mamba Transfer

## Motivation

Experiment 014 found that the existing teacher-guided objective can reduce attention-output NMSE while losing to an equal-budget CE-only Mamba control on held-out language loss. Experiment 015 tests the specific explanation suggested by the biology and accelerator-physics design review: **pointwise trajectory values are insufficient; a replacement must preserve the local interface response and must be gated in under a bounded system-level quality budget.**

This is a protocol for a new causal test. It does not assume that the proposed mechanism will succeed.

## Fixed Endpoint

The frozen source is `HuggingFaceTB/SmolLM-135M`. Attention layers 0 and 1 are replaced sequentially by Mamba mixers with state sizes 64 and 96. All source embeddings, MLPs, norms, output layers, remaining attention layers, and original attention modules held inside wrappers remain frozen. The endpoint has `α=1` for both replacement gates, so the two Mamba branches solely provide the former attention outputs. This remains a two-layer hybrid and is not a pure-Mamba conversion.

Every condition has identity-initialized rank-8 residual input and output adapters around each Mamba mixer:

> `M'_k(h) = (I + U_out V_outᵀ) M_k((I + U_in V_inᵀ)h)`

The adapters are trainable in every condition and have identical rank, parameter count, optimizer schedule, and initialization. Their purpose is to test interface matching rather than to give teacher-guided conditions hidden extra capacity.

## Three Matched Conditions

| Label | Optimization signal | Gate policy | Teacher output accessible to optimizer? |
|---|---|---|---|
| A — CE-adapter | Next-token CE only | Fixed common gate schedule | No |
| B — value/logit functional | Local attention-output value loss; teacher-logit KL; CE | Fixed common gate schedule | Yes |
| C — adaptive interface functional | Value/logit functional loss plus directional finite-difference operator loss | Bounded development-set gate controller | Yes |

All conditions use the same calibration texts, Mamba state sizes, adapter rank, layer order, maximum sequence length, optimizer-update ceiling, optimizer settings, prompts, seeds, and final test data. The random-number generator is reset before construction of each condition, so all three conditions begin from identical Mamba and adapter parameters within a seed.

## Interface Objectives

Let `A_k(h)` be the frozen source attention output, `M'_k(h)` the adapted Mamba output, and `T` the frozen source language model. The value objective is:

> `L_value = NMSE(M'_k(h), A_k(h)) + 0.5[1 − cosine(M'_k(h), A_k(h))] + 0.2[log(RMS(M'_k(h))/RMS(A_k(h)))]²`

For one deterministic seeded, unit-RMS perturbation `v` per calibration update, use `ε = 0.01 × RMS(h)` and finite differences:

> `L_tangent = ||[M'_k(h + εv) − M'_k(h)] − [A_k(h + εv) − A_k(h)]||² / (||A_k(h + εv) − A_k(h)||² + 1e−8)`

Condition B uses `L_value` but not `L_tangent`. Condition C uses `0.7 L_value + 0.3 L_tangent` in its local phase and `0.50 KL + 0.20 CE + 0.15 L_value + 0.15 L_tangent` in its gate phase. Condition B uses the Experiment 014 functional gate objective, `0.70 KL + 0.20 CE + 0.10 L_value`. Condition A uses CE only.

## Bounded Gate Controller for Condition C

Conditions A and B use the same fixed schedule: `α=0.05` for the first ten gate updates and then a linear rise to one. Condition C proposes gate targets `0.10, 0.20, …, 1.00` in order. At each target, it uses an equal share of the fixed gate-update budget and accepts the target only when both development diagnostics hold:

> `D(candidate) − D(last accepted) ≤ 0.05` nats/token  
> `KL(T logits || hybrid logits) ≤ 8.0`

where `D` is development CE loss measured on the fixed validation subset. If the target fails, the condition performs its remaining assigned stabilization updates at the previously accepted gate. The controller never inspects test data. If `α=1` is not reached at the end of the fixed budget, that seed is a **failed replacement** and is recorded as such; it is not rescued by forcing an uncontrolled final jump.

## Data, Budget, and Seeds

| Role | WikiText-2 raw partition | Size | Allowed use |
|---|---|---:|---|
| Calibration | Train | 1,024 usable sequences | Optimization inputs only |
| Development | Validation | 64 usable sequences | Diagnostics and condition-C gate control only |
| Final evaluation | Test | 128 usable sequences / fixed scoring tokens | Final endpoint evaluation and fixed-prompt continuations only |

Every condition receives 540 local updates and at most 180 gate updates: 720 updates per condition. No condition is allowed extra optimization updates. Initial replications will use five paired seeds: `20260841`, `20260842`, `20260843`, `20260844`, and `20260845`.

## Primary Outcome and Decision Rules

The primary outcome is final token-weighted causal-language-model loss on the untouched test sequences after both gates are at one. For a successful cross-architecture interface-transfer claim, condition C must satisfy all of the following:

| Criterion | Prespecified rule |
|---|---|
| Integrity | `α=0` reproduces frozen teacher test loss within `1e−5` in every condition and seed |
| Endpoint | Both C gates reach `α=1` in at least 4 of 5 seeds; failures count against C |
| Stability | Final losses are finite in all nonfailed runs |
| Primary causal comparison | C test loss is lower than A in at least 4 of 5 paired seeds and has mean `A − C ≥ 0.05` nats/token |
| Mechanism comparison | C test loss is lower than B in at least 4 of 5 paired seeds and has mean `B − C ≥ 0.03` nats/token |
| Interface diagnostic | C has lower development directional-response NMSE than B in at least 4 of 5 seeds |
| Scaling readiness | C mean test loss is no more than 0.15 nats/token above the frozen teacher and fixed-prompt English is coherent in at least 4 of 5 seeds |

A third attention replacement, a pure-Mamba claim, or any 7B experiment remains prohibited unless the primary causal comparison and scaling-readiness criteria all pass.

## Interpretation Guardrails

If C beats A but not B, the teacher’s trajectory values/logits help but directional matching and adaptive gating do not add value. If C beats both A and B, the data support the narrow hypothesis that local interface geometry and staged bounded replacement add transferable information. If A remains best, then the teacher-guided approaches tested here are not justified at the present data and compute scale. No outcome permits a claim that full pretraining is generally unnecessary without further controls and full-architecture evidence.
