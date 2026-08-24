# Experiment 020B Protocol: Relational Geometry at the Deployed Two-Layer Interface

**Status:** Pre-registered endpoint protocol. Implementation and endpoint execution are prohibited until this file is committed unchanged.

## 1. Objective

Experiment 020A was a development-only coefficient-selection pilot. Under its locked rule, \(\lambda=0.20\) was selected as the largest relation-geometry coefficient that reduced mean development relation KL while remaining within the development CE safety ceiling. Experiment 020B asks the causal endpoint question that the pilot could not answer:

> Does the **correctly indexed** frozen-teacher post-block token-relation target improve a fully deployed two-layer Transformer–Mamba hybrid on a new fresh held-out language-model slice beyond both equal-budget conditional CE transport and an equal-compute topology-destroyed relational control?

The experiment is a frozen-hybrid study using `HuggingFaceTB/SmolLM-135M`. It does not test direct weight transplantation, standalone Mamba conversion, a third replacement layer, 1B/7B scaling, 4-bit quantization, or teacher use at inference.

A lower local relation loss, a lower output entropy, or fluent prompts cannot establish success. The primary endpoint is fresh token-weighted next-token loss.

## 2. Frozen Endpoint and Shared Training

Attention in layers 0 and 1 is sequentially replaced by fresh `MambaMixer` modules with state sizes 64 and 96. The deployed conditional transport map is

> \(T_\theta(h,m)=m+W_{up}\operatorname{SiLU}(W_{down}[\operatorname{RMS}(h);\operatorname{RMS}(m);\operatorname{RMS}(h)-\operatorname{RMS}(m)])\),

with rank 16 and exactly zero-initialized \(W_{up}\). The replacement is

> \(y=(1-\alpha)A(h)+\alpha T_\theta(h,M_\phi(h))\).

All source weights—embeddings, norms, MLPs, output head, unwrapped attention layers, and frozen teacher—remain frozen. Only the two fresh Mamba mixers and their rank-16 conditional transports may train. At \(\alpha=0\), frozen source behavior must be exact. At the final endpoint, both active gates must equal one.

Every condition receives 720 updates: 300 for layer 0 and 420 for layer 1. Alpha is 0.05 for the first 10 updates of each active layer and increases linearly to 1.0 thereafter; completed earlier layers remain at 1.0. AdamW uses learning rate \(10^{-4}\), weight decay 0.01, and gradient-norm clipping at 1.0. Each update uses one calibration sequence.

## 3. Correct and Topology-Destroyed Relation Objectives

Let \(H_T\) and \(H_S\) be frozen-teacher and hybrid hidden states at the input of decoder layer 2 after both replacement sites in the full sequence computation. Excluding the terminal input position, form token-normalized pairwise similarity matrices and row-wise distributions:

> \(R(H)_{ij}=\operatorname{softmax}_{j}(\operatorname{RMS}(H)_i\operatorname{RMS}(H)_j^\top/(d\tau_r))\), with \(\tau_r=0.20\).

The correctly indexed target is

> \(L_{rel}=\frac{1}{L'}\sum_i\operatorname{KL}(R(H_T)_i\parallel R(H_S)_i)\).

The topology-destroyed target uses the same teacher relation values, coefficient, shape, compute, and row-wise normalization but cyclically shifts teacher target columns by \(s=\lfloor L'/2\rfloor\) before comparison:

> \(L_{perm}=\frac{1}{L'}\sum_i\operatorname{KL}(\operatorname{roll}(R(H_T)_i,s)\parallel R(H_S)_i)\).

The shift leaves the marginal target values available but breaks their correspondence to the student sequence positions. It is not expected to be useful; it is an equal-compute causal control for whether an apparent benefit depends on the original teacher relational topology rather than the mere presence of an extra auxiliary loss.

| Condition | Objective | Teacher-derived optimizer signal | Causal role |
|---|---|---|---|
| **A — Conditional CE transport** | \(L_A=CE\) | None | Controls ordinary end-to-end hybrid adaptation. |
| **B — Correct relational geometry** | \(L_B=CE+0.20L_{rel}\) | Correct post-block teacher token-relation topology | Tests the candidate teacher-specific mechanism. |
| **C — Permuted relational control** | \(L_C=CE+0.20L_{perm}\) | Same relation values with position topology destroyed | Controls equal auxiliary computation and teacher-value exposure. |

Every condition has identical paired initialization, Mamba state sizes, transport rank, parameter count, trainable parameters, optimizer, alpha schedule, update count, calibration sequence order, prompts, and evaluation procedure. The only variation is the declared loss. The teacher is never used at inference.

## 4. Data and Strict Isolation

| Role | WikiText-2 raw split | Eligible sequences | Use |
|---|---|---:|---|
| Calibration | Train | First 1,024 | Optimization only. |
| Development | Validation | 65–128 | Fixed diagnostics only; never used to alter this protocol. |
| Fresh final evaluation | Test | 513–640 | Loaded, tokenized, and scored only after all three conditions complete all 720 updates in a seed. |

Experiment 014–015 used test sequences 1–128, Experiment 017 used 129–256, Experiment 018 used 257–384, and Experiment 019 used 385–512. Experiment 020B must skip the first 512 eligible test sequences. No condition, script, log, or pre-execution diagnostic may request the test split before the final endpoint process.

The five paired seeds are `20260891`, `20260892`, `20260893`, `20260894`, and `20260895`. No architecture, rank, coefficient, temperature, target shift, update budget, optimizer setting, alpha schedule, data offset, sequence count, seed, endpoint, prompt, or acceptance rule may change after seed 20260891 begins.

## 5. Measurements

| Metric | Partition | Status |
|---|---|---|
| Token-weighted next-token loss and perplexity | Fresh final slice | **Primary endpoint** |
| Teacher-relative fresh-loss gap | Fresh final slice | Scale-readiness metric |
| Correct relation KL and permuted relation KL | Development | Mechanism diagnostics only |
| Token-normalized source-logit KL | Development | Diagnostic only |
| Post-layer-1 and post-layer-2 residual drift | Development | Amplification diagnostic only |
| Transport correction RMS ratio, effective rank, and gradient allocation | Development/training traces | Capacity-use diagnostics only |
| Token entropy and 20-bin top-token ECE | Fresh final slice | Confidence diagnostics only |
| Fixed greedy continuations | Final alpha-one endpoint | Coherence sanity check only |

Each condition is evaluated at alpha zero on the fresh final slice after all training, only to verify exact source preservation. The fresh primary endpoint is evaluated at alpha one.

## 6. Prespecified Acceptance Criteria

| Category | Rule |
|---|---|
| Integrity | Every condition reproduces frozen teacher fresh-test loss at alpha zero within \(10^{-5}\) in all five seeds. |
| Endpoint | Both gates equal alpha one in every condition in at least 4/5 seeds; incomplete endpoints fail. |
| Stability | Every valid final loss is finite. |
| Correct-topology benefit versus CE | B beats A in at least 4/5 paired seeds and mean \(A-B\geq0.03\) nats/token. |
| Correct-topology benefit versus permuted control | B beats C in at least 4/5 paired seeds and mean \(C-B\geq0.03\) nats/token. |
| Topology interpretation | The observed B advantage must not be explained by a lower permuted-control loss; C is an equal-compute control rather than a transfer success candidate. |
| Scale readiness | B is within 0.15 nats/token of the frozen teacher, fresh ECE does not exceed A by more than 0.01, and coherent continuations occur in at least 4/5 seeds. |
| Third-layer / scaling authorization | Every prior criterion passes. |

A result where B lowers development relation KL but fails fresh loss or either matched causal comparison is a negative result. A result where B and C are equivalent is not evidence that correct teacher topology transferred. A result where C outperforms B rejects the stated relational-correspondence hypothesis under this fixed protocol.

## 7. Reporting Constraints

No result may be called successful transfer merely because it improves a relation diagnostic, local drift, output entropy, calibration, or prompt appearance. Reporting must distinguish the shared pretrained frozen backbone from the two fresh Mamba branches. It must not claim that pretraining is unnecessary or that the hybrid is a pure Mamba model unless the stated paired fresh-test criteria pass.

Any runtime repair is permitted only when it prevents execution while leaving the declared mathematics, data isolation, and all acceptance rules unchanged. A repair must be documented before rerunning seed 20260891.
