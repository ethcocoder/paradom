# Experiment 017: Featurewise Moment Calibration for a Two-Layer Transformer-to-Mamba Interface

## Motivation

Experiment 016 found that the layer-1 `α=0.80 → 0.90` failure was smooth rather than discontinuous. A static Mamba-versus-attention distribution mismatch—especially in featurewise variance geometry—was progressively exposed to the frozen downstream Transformer as alpha increased. Experiment 017 tests one narrow causal hypothesis: a featurewise affine Mamba-output calibrator trained to match source-attention moments can improve the complete two-layer endpoint beyond equal-capacity CE-only and value-functional controls.

This protocol does not assume the hypothesis is correct. It is a fresh causal test with a newly reserved final evaluation slice.

## Fixed Architecture and Calibrator

The frozen source is `HuggingFaceTB/SmolLM-135M`. Attention layers 0 and 1 are sequentially replaced by fresh Mamba mixers with state sizes 64 and 96. Source embeddings, MLPs, norms, output head, remaining attention layers, and original source attention retained within wrappers remain frozen.

Every condition has the same trainable modules at each replacement layer:

> `M_cal(h) = gamma ⊙ [B_out M(B_in h)] + beta`

where `B_in` and `B_out` are the rank-8 identity-initialized residual adapters used in Experiment 015, `gamma` is a per-hidden-feature scale initialized to one, and `beta` is a per-hidden-feature shift initialized to zero. The calibrator and low-rank adapters are present in every condition. The only causal difference is the optimization objective.

## Three Equal-Parameter Conditions

| Label | Local phase | Gate phase | Teacher-derived signal in optimizer? |
|---|---|---|---|
| A — CE-calibrator | Next-token CE at `α=1` | CE only | No |
| B — value-functional calibrator | Source-attention value loss | Token-normalized source-logit KL + CE + value | Yes |
| C — moment-functional calibrator | Value loss + featurewise moment loss | Token-normalized source-logit KL + CE + value + moment | Yes |

The moment loss is computed only on calibration sequences:

> `L_moment = mean((mean_t(M_cal) − mean_t(A))² / (var_t(A)+eps)) + mean((log(var_t(M_cal)+eps) − log(var_t(A)+eps))²)`

where mean and variance are over batch and sequence positions for each hidden feature. Condition B has the same calibrator but does not optimize `L_moment`; condition A has the same calibrator but receives no teacher branch or logits in its objective.

All three conditions begin from identical Mamba, adapter, and calibrator parameters within each paired seed. They use the same data, layer order, maximum length, number of trainable parameters, optimizer-update ceiling, learning rates, gate targets, and prompts.

## Losses and Gate Controller

Condition B uses `L_value` during local fitting. Condition C uses `0.65 L_value + 0.35 L_moment`. During gates, B uses `0.70 KL_token + 0.20 CE + 0.10 L_value`; C uses `0.55 KL_token + 0.20 CE + 0.10 L_value + 0.15 L_moment`. Condition A uses CE only.

`KL_token` is temperature-scaled KL summed over all vocabulary-position entries and divided by the number of predicted tokens. This is explicitly different from the legacy batchmean KL retained only for diagnostic continuity with Experiment 015.

All conditions use the common fixed gate schedule `α=0.05` for the first ten gate updates and then a linear increase to one. Condition C additionally records, but does not use to change its update budget, a development-only safety trace at targets `0.10, 0.20, …, 1.00`. Its endpoint is considered valid only if both replacement gates reach one. The trace reports token-normalized KL; a value above 0.22 is flagged but does not permit a post-hoc objective change.

## Data Isolation, Budget, and Seeds

| Role | Data | Quantity | Permitted use |
|---|---|---:|---|
| Calibration | WikiText-2 raw train | First 1,024 eligible sequences | Optimization only |
| Development | WikiText-2 raw validation | First 64 eligible sequences | Diagnostics only; no test selection |
| Fresh final evaluation | WikiText-2 raw test | Eligible sequences 129–256, 128 sequences | Final loss and fixed-prompt quality only |

The final evaluation slice is fresh for Experiment 017: Experiments 014 and 015 used the first 128 eligible WikiText-2 test sequences, while Experiment 017 skips those sequences. No condition may inspect this fresh slice before the final endpoint evaluation.

Each condition receives 540 local plus 180 gate updates: 720 updates total. Seeds are `20260851`, `20260852`, `20260853`, `20260854`, and `20260855`.

## Acceptance and Scaling Rules

| Criterion | Prespecified rule |
|---|---|
| Integrity | `α=0` reproduces the frozen source on the fresh final slice within `1e−5` in every condition and seed |
| Endpoint | Both gates reach `α=1` in all three conditions in at least 4/5 seeds; failures count against the condition |
| Stability | All completed final losses are finite |
| Primary causal comparison | C final test loss is lower than A in at least 4/5 paired seeds and mean `A − C ≥ 0.05` nats/token |
| Mechanism comparison | C final test loss is lower than B in at least 4/5 paired seeds and mean `B − C ≥ 0.03` nats/token |
| Diagnostic mechanism check | C improves layer-1 development featurewise log-variance mismatch over B in at least 4/5 seeds |
| Scaling readiness | C mean final loss is within 0.15 nats/token of the frozen teacher and English continuations are coherent in at least 4/5 seeds |

No third attention replacement, pure-Mamba conversion, 7B scale-up, or claim that pretraining is unnecessary is allowed unless all relevant criteria pass.

## Interpretation Rules

If C beats A and B, the evidence supports the narrow claim that source-attention featurewise moment calibration adds useful cross-architecture supervision beyond equal-capacity CE adaptation. If B beats A but C does not beat B, teacher value/logit signals help but the proposed moment mechanism adds nothing. If A remains best, the current teacher-guided objectives remain unsupported at this scale. All outcomes are limited to the tested hybrid endpoint and corpus protocol.
