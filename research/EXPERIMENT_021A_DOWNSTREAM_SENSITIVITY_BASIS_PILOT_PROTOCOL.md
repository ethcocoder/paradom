# Experiment 021A Protocol: Development-Only Downstream-Sensitivity Basis Pilot

**Status:** Preregistered. This file must be committed before implementation. No final-test split may be requested, loaded, tokenized, inspected, or scored by this pilot.

## 1. Research Question

The repeated Paradom failure mode is a local-to-global mismatch: attention values, directions, moments, relation topology, output sharpness, and a downstream activation anchor can improve their own teacher-based diagnostics without beating equal-budget CE adaptation at a fresh language-model endpoint. Experiment 021A tests a structurally different preliminary question:

> Can a fixed basis derived from the frozen downstream sensitivity of each residual-stream coordinate allocate a rank-16 transport correction in a way that is CE-safe and distinguishable from an equal-spectrum, topology-destroyed basis?

The pilot is not an endpoint experiment. It cannot establish knowledge transfer, generalization, a successful Transformer-to-Mamba conversion, or permission to scale.

## 2. Frozen Hybrid and Common Training

The frozen checkpoint is `HuggingFaceTB/SmolLM-135M`. Attention in layers 0 and 1 is sequentially wrapped with fresh `MambaMixer` modules of state sizes 64 and 96 and deployed rank-16 conditional transport:

> \(y=(1-\alpha)A(h)+\alpha T_q(h,M(h))\),

> \(T_q(h,m)=m+q\odot W_{up}\operatorname{SiLU}(W_{down}[\operatorname{RMS}(h);\operatorname{RMS}(m);\operatorname{RMS}(h)-\operatorname{RMS}(m)]).\)

The rank is 16. \(W_{up}\) is initialized exactly to zero. At alpha zero, frozen source attention must be exactly reproduced. The source embeddings, norms, MLPs, output head, unwrapped attention layers, and teacher remain frozen. Only the fresh Mamba mixers and rank-16 transport weights train.

Each condition receives 300 layer-0 updates and 420 layer-1 updates, one calibration sequence per update, using AdamW (learning rate \(10^{-4}\), weight decay 0.01, gradient-norm clipping 1.0). The active gate is 0.05 for its first 10 updates and then rises linearly to 1.0. Earlier completed layers stay at alpha one.

The primary training objective is **next-token CE only**. The teacher has no activation, logit, relation, entropy, gradient, or other optimizer loss term and is never used at inference.

## 3. Fixed Sensitivity-Basis Derivation

Before any branch training in a seed, the frozen teacher runs on the first 128 eligible calibration sequences. For each active layer \(\ell\), let \(a_\ell\) be the frozen attention output and let \(\ell_{CE}\) be the teacher’s causal CE. Compute

> \(f_{\ell,j}=\operatorname{mean}_{x,t}[(\partial \ell_{CE}(x)/\partial a_{\ell,t,j})^2]\).

The correct fixed gain is

> \(q_{\ell,j}=\operatorname{RenormMean1}[\operatorname{clip}(\sqrt{f_{\ell,j}/(\operatorname{mean}(f_\ell)+10^{-12})},0.50,2.00)].\)

The square root is fixed before execution; no exponent, clipping bound, rank, state size, or other hyperparameter may be tuned in this pilot. The basis is fixed after derivation and receives no gradient.

A cyclic permutation by \(s=\lfloor d/2\rfloor\) creates the topology-destroyed control \(q^{perm}=\operatorname{roll}(q,s)\). It preserves every gain value, mean, variance, bounds, transport rank, parameter count, initialization, compute, and CE objective while breaking correspondence to the frozen residual-stream coordinates that produced the sensitivity estimate.

## 4. Conditions

| Condition | Correction gain \(q\) | Teacher-derived optimizer signal | Causal role |
|---|---|---|---|
| **A — Unit conditional CE transport** | All ones | None | CE-only architecture baseline. |
| **B — Correct sensitivity-basis CE transport** | Correctly ordered fixed \(q\) | None | Tests sensitivity-informed capacity allocation. |
| **C — Permuted sensitivity-basis CE transport** | Cyclically permuted fixed \(q\) | None | Equal-spectrum, topology-destroyed causal control. |

All three conditions share within-seed fresh Mamba initialization, transport initialization, rank, trainable parameter count, source weights, optimizer, alpha schedule, update budget, calibration order, development data, and diagnostics. No condition is allowed extra CE updates or source-gradient calculations. The one shared frozen-teacher basis-derivation pass occurs before all conditions and is not an optimizer update.

## 5. Development-Only Data and Prohibition

| Role | WikiText-2 raw split | Eligible sequences | Permitted use |
|---|---|---:|---|
| Sensitivity derivation | Train | First 128 | Frozen teacher gradients only, before branch training. |
| Calibration | Train | First 1,024 | CE optimization only. |
| Development diagnostics | Validation | 65–128 | Recorded after alpha-one endpoint only. |
| Test split | Test | None | **Prohibited.** |

The pilot seeds are `20260901`, `20260902`, `20260903`, `20260904`, and `20260905`. The implementation must record `test_split_requested: false` for every completed seed. No task may alter the protocol after seed 20260901 starts.

## 6. Development Diagnostics

The diagnostics are engineering checks and cannot be interpreted as endpoint success.

| Diagnostic | Definition | Purpose only |
|---|---|---|
| Development CE | Token-weighted next-token CE | Safety relative to A. |
| Sensitivity-weighted attention-output discrepancy | Mean across active layers of \(\operatorname{mean}_{t,j}[f_{\ell,j}(T_q-A)^2]/\operatorname{mean}(f_\ell)\) | Checks whether the basis changes the error in frozen downstream-sensitive coordinates. |
| Unweighted attention-output MSE | Mean across active layers of \(\operatorname{mean}_{t,j}(T_q-A)^2\) | Distinguishes generic magnitude changes. |
| Post-layer-1 and post-layer-2 drift | RMS relative to frozen teacher | Amplification diagnostic. |
| Basis summaries | Min, max, mean, SD, cyclic shift | Verifies matching spectrum and ordering intervention. |
| Correction RMS ratio and effective rank | Transport correction diagnostics | Capacity-use check. |

## 7. Locked Feasibility Rule

A separate endpoint protocol is allowed only if all conditions hold:

| Requirement | Rule |
|---|---|
| Source integrity | Alpha-zero development CE differs from teacher by at most \(10^{-5}\) in every condition and seed. |
| Completion and stability | Both gates equal one and all endpoint metrics are finite in every condition and seed. |
| Test isolation | `test_split_requested: false` in all five results. |
| CE safety | Mean development CE of B is no more than 0.020 nats/token above A. |
| Correct-basis diagnostic effect versus A | B has a lower mean sensitivity-weighted discrepancy than A. |
| Correct-basis diagnostic effect versus C | B has a lower mean sensitivity-weighted discrepancy than C. |

If any requirement fails, no test-split endpoint is permitted for this mechanism. If all pass, Experiment 021B must use the exact fixed basis and compare A, B, and C on a new untouched final slice; this pilot may not be used to select or alter the gain formula.

## 8. Reporting Boundary

A development-only difference, a lower sensitivity-weighted discrepancy, or a coherent prompt cannot prove checkpoint knowledge transfer. The report must describe any result as a feasibility diagnostic and must state that B needs to beat both A and C on a new fresh final endpoint before it is interpreted as a teacher-specific mechanism.
