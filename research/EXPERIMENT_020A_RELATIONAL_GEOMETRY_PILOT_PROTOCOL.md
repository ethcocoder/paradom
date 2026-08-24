# Experiment 020A Protocol: Development-Only Relational-Geometry Pilot

**Status:** Preregistered design. This pilot may be implemented only after this file is committed unchanged. It is a train-and-validation diagnostic study, not an endpoint experiment.

## 1. Purpose and Scope

Experiments 013–019 show that local attention imitation, directional matching, moments, a downstream anchor, and output sharpening do not establish a practically useful teacher-specific transfer advantage beyond paired CE adaptation. The remaining mechanism hypothesis is that the frozen downstream Transformer is sensitive to the **token-to-token residual geometry** of the deployed hybrid after both active Mamba transport layers, rather than to any individual hidden coordinate or output-distribution sharpness.

Experiment 020A has one deliberately limited purpose: select one numerically stable relational-loss coefficient on development data before a future endpoint study is written. It must not request, load, tokenize, inspect, score, or otherwise access WikiText-2 test data. It does not test a third replacement layer, a standalone Mamba model, 1B/7B scaling, or quantization.

> **Not a success criterion:** A lower development relation loss, a lower entropy, a lower development CE, or coherent text produced during this pilot does not establish language-model transfer. The pilot only locks one coefficient and verifies that the mechanism can be trained without violating integrity conditions.

## 2. Frozen Hybrid and Shared Training Setup

The source is `HuggingFaceTB/SmolLM-135M`. Attention in layers 0 and 1 is replaced sequentially by fresh `MambaMixer` modules with state sizes 64 and 96. Each Mamba mixer is followed by the deployed conditional residual transport map from Experiment 018:

> \(T_\theta(h,m)=m+W_{up}\operatorname{SiLU}(W_{down}[\operatorname{RMS}(h);\operatorname{RMS}(m);\operatorname{RMS}(h)-\operatorname{RMS}(m)])\).

The transport rank is 16 and \(W_{up}\) is initialized to exact zero. The attention replacement uses

> \(y=(1-\alpha)A(h)+\alpha T_\theta(h,M_\phi(h))\).

All source weights, including embeddings, norms, MLPs, output head, unwrapped layers, and frozen teacher are frozen. Only both fresh Mamba mixers and their rank-16 transport maps may be trained. At \(\alpha=0\), the source behavior must be exactly reproduced. The complete deployment endpoint requires both active gates at \(\alpha=1\).

Every condition receives the same 720 updates: 300 while activating layer 0, then 420 while activating layer 1. Alpha is 0.05 for the first 10 updates of a layer and increases linearly to 1.0 thereafter. AdamW uses learning rate \(10^{-4}\), weight decay 0.01, and gradient-norm clipping at 1.0. Each update uses one calibration sequence.

## 3. Post-Block Token-Relation Auxiliary

Let \(H_T\in\mathbb{R}^{L'\times d}\) and \(H_S\in\mathbb{R}^{L'\times d}\) be, respectively, the frozen teacher and hybrid hidden-state inputs to **decoder layer 2** (zero-indexed), restricted to nonterminal sequence positions \(L'=L-1\). This is a common location after both active substitutions and the frozen MLP/residual computations of layers 0 and 1.

For a hidden-state sequence \(H\), define token-normalized features \(\bar H=\operatorname{RMS}(H)\), token cosine similarities \(S(H)=\bar H\bar H^\top/d\), and a row-wise token-relation distribution

> \(R(H)_{ij}=\operatorname{softmax}_{j}(S(H)_{ij}/\tau_r)\), where \(\tau_r=0.20\).

The relational auxiliary is

> \(L_{rel}=\frac{1}{L'}\sum_i\operatorname{KL}(R(H_T)_i\parallel R(H_S)_i)\).

Teacher representations are detached. There is no learned projection head, no target token permutation, no memory bank, no teacher use at inference, and no teacher-logit, attention-value, moment, or direct output-entropy objective in this pilot. This loss is intentionally post-deployment and relational; it is not a new version of direct local value matching.

## 4. Pilot Conditions and Paired Coefficients

The fixed paired seeds are `20260881`, `20260882`, `20260883`, `20260884`, and `20260885`. Within a seed, every condition must begin with identical random Mamba and transport initialization, calibration order, data, schedule, parameter count, and endpoint. The sole change is the relational coefficient.

| Condition | Objective | Role |
|---|---|---|
| **A — Conditional CE transport** | \(L_A=CE\) | Required CE-only baseline. |
| **B05 — Relational pilot 0.05** | \(L_{B05}=CE+0.05L_{rel}\) | Low auxiliary strength. |
| **B10 — Relational pilot 0.10** | \(L_{B10}=CE+0.10L_{rel}\) | Middle auxiliary strength. |
| **B20 — Relational pilot 0.20** | \(L_{B20}=CE+0.20L_{rel}\) | High auxiliary strength. |

No coefficient may be added, removed, altered, or retried after any seed begins. The frozen teacher may be run only to supply \(H_T\) during optimization and to evaluate the validation alpha-zero integrity check.

## 5. Data Isolation

| Role | WikiText-2 raw split | Eligible sequences | Permitted use |
|---|---|---:|---|
| Calibration | Train | First 1,024 | Optimization only. |
| Development | Validation | Eligible sequences 65–128 | All pilot diagnostics and coefficient selection only. |
| Test | Test | None | **Prohibited.** The split must not be requested, loaded, tokenized, inspected, or scored. |

The pilot implementation must contain no `split="test"` call and must write `test_split_requested: false` into each results file. This is an integrity requirement, not merely a convention.

## 6. Measurements

All measurements are development-only and are descriptive except the locked coefficient-selection rule.

| Measurement | Role |
|---|---|
| Token-weighted next-token CE | Development endpoint and safety diagnostic. |
| Token-normalized source-logit KL | Development diagnostic; not a cross-context primary endpoint. |
| Post-block relation KL \(L_{rel}\) | Mechanism diagnostic. |
| Post-layer-1 and post-layer-2 residual drift to frozen teacher | Amplification diagnostic. |
| Per-layer transport correction RMS/Mamba RMS and singular-value effective rank | Capacity-use diagnostic. |
| Mamba versus transport gradient RMS ratio during optimization | Gradient-allocation diagnostic. |
| Alpha-zero source-loss deviation on validation | Integrity diagnostic. |

No generations and no validation quantity may select an endpoint mechanism beyond the locked coefficient rule below.

## 7. Locked Coefficient-Selection Rule

First compute, across the five paired seeds, each B condition’s mean development CE difference relative to A and mean relational-loss difference relative to A. A B condition is **eligible** only if all of the following are true:

1. all final development CEs and relation losses are finite in all five seeds;
2. both gates equal one at its endpoint in all five seeds;
3. its maximum validation alpha-zero source-loss deviation is at most \(10^{-5}\);
4. its mean development CE is no more than 0.020 nats/token higher than A; and
5. its mean relation loss is strictly lower than A.

Among eligible coefficients, select the **largest coefficient**. If no coefficient is eligible, Experiment 020 endpoint work is cancelled and no test partition may be accessed. The selected coefficient is a development-only choice; it is not evidence of a fresh-held-out benefit.

## 8. Reporting and Prohibitions

A pilot report must present every coefficient, every seed, the integrity outcomes, and the locked selection calculation. It must use the language “development-only” throughout and must not claim a transfer win, scale readiness, or model conversion. No fresh test slice may be used to decide whether the selected coefficient is acceptable.

The only authorized next action after a successful pilot report is a separate committed endpoint protocol with a new, unused test range and a CE baseline plus an equal-compute topology-destroyed relational control. Implementation of an endpoint experiment before that protocol is forbidden.
