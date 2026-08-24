# Experiment 019 Protocol: Teacher-Sharpened Output Fine-Tuning

**Status:** Pre-registered protocol. Implementation and seed execution are prohibited until this document is committed unchanged.

## 1. Objective

Experiment 019 tests the user-proposed idea that, after a two-layer Transformer-to-Mamba hybrid has been trained to the deployed `alpha=1` endpoint, a new **output-sharpening fine-tuning** stage can improve its language output. The question is not whether a lower-entropy distribution looks more decisive. The causal question is whether a teacher-sharpened output target improves the hybrid beyond both matched ordinary CE continuation and non-teacher self-entropy sharpening.

The experiment remains a **two-layer frozen-hybrid** study. It does not test a standalone Mamba conversion, direct static weight transfer, a third replacement layer, a 1B/7B model, or quantization.

> **Hypothesis:** A low-temperature distribution from the frozen Transformer teacher, used only during a short deployed output-level fine-tuning stage, will sharpen useful token preferences and yield lower fresh held-out language loss than equal-budget CE continuation and self-entropy sharpening.

A result that merely reduces output entropy, gives fluent prompts, or improves over the pre-stage but does not beat both controls is not evidence for teacher-specific knowledge transfer.

## 2. Frozen Endpoint and Shared Pre-Stage

The source is `HuggingFaceTB/SmolLM-135M`. Layers 0 and 1 are sequentially replaced with fresh Mamba mixers of state sizes 64 and 96. The deployed wrapper is the **conditional residual interface transport map** from Experiment 018:

> `y = (1 - alpha) A(h) + alpha T_theta(h, M_phi(h))`

where `A` is frozen source attention, `M_phi` is a fresh Mamba mixer, and

> `T_theta(h,m) = m + W_up SiLU(W_down concat(RMS(h), RMS(m), RMS(h) - RMS(m)))`.

The transport bottleneck rank is 16, `W_up` is exactly zero initialized, and all source weights—embeddings, norms, MLPs, output head, teacher, and unwrapped layers—remain frozen. At `alpha=0` the source output must be exactly preserved; at the final endpoint both active gates must equal one.

Each condition begins from the same **pre-stage**: 720 updates of conditional CE transport, comprising 300 updates at layer 0 and 420 updates at layer 1. Alpha is 0.05 for the first 10 updates of each layer and rises linearly to 1.0 for the rest; layer 0 stays at one while layer 1 trains. This establishes the same fully deployed hybrid before the output-sharpening comparison.

## 3. Matched Output Fine-Tuning Conditions

After the pre-stage, all conditions run a short 180-update output-stage at both gates `alpha=1.0`. The only trainable parameters throughout are the two fresh Mamba mixers and their rank-16 deployed conditional transport maps. The output head is frozen in every condition.

Let `p_S = softmax(z_S)` denote student hybrid token probabilities and `p_T^tau = softmax(z_T / tau)` denote frozen teacher probabilities at temperature `tau = 0.7`. Let `H(p_S) = -sum_v p_S(v) log p_S(v)` averaged over nonterminal sequence positions. Let token CE use the real next token.

| Condition | Output-stage objective | Teacher-derived optimizer signal | Causal role |
|---|---|---|---|
| **A — CE continuation** | `L_A = CE` | None | Controls the benefit of extra ordinary fine-tuning and total update budget |
| **B — Self-entropy sharpening** | `L_B = CE + 0.05 × H(p_S)` | None | Controls non-teacher output concentration at the same deployed endpoint |
| **C — Teacher-temperature sharpening** | `L_C = CE + 0.20 × KL(p_T^0.7 || p_S)` | Low-temperature teacher logits only | Tests whether teacher-specific output preference improves the transferred hybrid |

All three conditions have identical paired initialization, pre-stage, Mamba state sizes, rank, parameter count, optimizer, trainable parameters, calibration sequence order, alpha schedule, output-stage step count, prompts, and final-evaluation procedure. The only variation is the declared output-stage loss.

The output stage uses AdamW with learning rate `5e-5`, weight decay `0.01`, gradient-norm clipping at 1.0, and one sequence per update. No logit temperature is used at inference. The teacher is never used at inference.

## 4. Data and Strict Isolation

| Role | WikiText-2 raw split | Eligible sequences | Use |
|---|---|---:|---|
| Calibration | Train | First 1,024 | Pre-stage and output-stage optimization only |
| Development | Validation | 65–128 | Fixed diagnostics only; no post-seed selection or schedule changes |
| Fresh final evaluation | Test | 385–512 | Loaded, tokenized, and scored only after all three conditions complete all updates in a seed |

Test sequences 1–128 were used by Experiments 014–015, 129–256 by Experiment 017, and 257–384 by Experiment 018. Experiment 019 skips the first 384 eligible test sequences. It may not load or inspect sequences 385–512 before every condition in the active paired seed completes all 900 updates.

The five paired seeds are `20260871`, `20260872`, `20260873`, `20260874`, and `20260875`. Each condition receives exactly 900 optimization updates: 720 common pre-stage updates and 180 output-stage updates. No loss weight, temperature, rank, learning rate, data offset, sequence count, seed, prompt, or endpoint definition may change after seed 20260871 begins.

## 5. Measurements

The primary endpoint is token-weighted next-token loss on the fresh final slice at both gates `alpha=1`. Every condition is separately evaluated at `alpha=0` on that same slice, after all training, solely to prove source preservation.

| Metric | Data partition | Status |
|---|---|---|
| Token-weighted next-token loss and perplexity | Fresh final slice | **Primary endpoint** |
| Teacher-relative fresh-test loss gap | Fresh final slice | Scale-readiness metric |
| Token entropy | Development and fresh final | Output-sharpening diagnostic, not a quality endpoint |
| Fixed-bin top-token ECE (20 equal-width confidence bins) | Development and fresh final | Confidence diagnostic, not a quality endpoint |
| Teacher-logit KL | Development only | Diagnostic, not optimization target in A/B |
| Fixed greedy continuations | Fresh final endpoint | Coherence sanity check |

For ECE, each next-token prediction contributes its maximum predicted probability as confidence and an indicator that its argmax token equals the observed next token. Bins with no observations are ignored; ECE is the token-count-weighted absolute confidence-minus-accuracy difference across nonempty bins.

## 6. Prespecified Acceptance Criteria

| Category | Rule |
|---|---|
| Integrity | Every condition reproduces frozen teacher fresh-test loss at `alpha=0` within `1e-5` in all five seeds |
| Endpoint | Both gates reach `alpha=1` in every condition for at least 4/5 seeds; incomplete endpoints count as failures |
| Stability | All valid final losses are finite |
| Sharpening behavior | B and C lower mean fresh-test entropy than A; this criterion is descriptive and cannot compensate for worse loss |
| Teacher-specific output benefit | C beats A and B on fresh test in at least 4/5 paired seeds and mean `A - C >= 0.03` and `B - C >= 0.03` nats/token |
| Scale readiness | C is within 0.15 nats/token of frozen teacher, fresh ECE does not exceed A by more than 0.01, and coherent continuations occur in at least 4/5 seeds |
| Third-layer / scaling authorization | All prior criteria pass |

A result where C decreases entropy but fails held-out loss, teacher-specific comparisons, or calibration is a negative result. A result where B and C both improve over A but C does not beat B is evidence only for generic output sharpening. A result where A remains best rejects this sharpening mechanism under the fixed protocol.

## 7. Prohibited Changes and Reporting Rules

The following are prohibited after seed 20260871 begins: changing architecture, state sizes, rank, pre-stage, output-stage losses, coefficients, teacher temperature, optimizer, updates, alpha schedule, calibration data, development offset, fresh-test offset, seeds, endpoint definition, prompts, or acceptance criteria.

A runtime failure can be fixed only when it prevents execution and the repair leaves the declared mathematics unchanged. Any repair must be documented before rerunning seed 20260871. Final reporting must distinguish lower entropy, generic CE fine-tuning, generic self-sharpening, teacher-specific output benefit, and complete architecture-transfer evidence. It must not call a hybrid a pure Mamba model or claim pretraining is unnecessary unless the stated causal criteria pass.

## 8. Status

This protocol authorizes future implementation and one first-seed validation only after it is committed unchanged. It does not authorize third-layer replacement, 1B/7B scale-up, 4-bit quantization, standalone Mamba claims, or test access outside the final endpoint process.
