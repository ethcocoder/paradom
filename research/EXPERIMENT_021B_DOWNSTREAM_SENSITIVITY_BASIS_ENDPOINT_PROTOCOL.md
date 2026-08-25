# Experiment 021B Protocol: Downstream-Sensitivity Basis at the Two-Layer Endpoint

**Status:** Pre-registered endpoint protocol. Implementation and final-slice execution are prohibited until this file is committed unchanged.

## 1. Objective

Experiment 021A was a development-only feasibility audit. It found that a fixed correction basis derived from the frozen teacher’s downstream residual-coordinate sensitivity was CE-safe and reduced a sensitivity-weighted development discrepancy relative to both unit and equal-spectrum permuted bases. It did not load or inspect the test split.

Experiment 021B asks the causal endpoint question:

> Does the **correctly ordered fixed downstream-sensitivity basis** improve a fully deployed two-layer Transformer–Mamba hybrid on a new fresh language-model slice beyond both unit-gain conditional CE transport and an equal-spectrum, topology-destroyed sensitivity-basis control?

This is a frozen-hybrid experiment using `HuggingFaceTB/SmolLM-135M`. It does not test direct static weight mapping, functional activation matching, output sharpening, relational topology, a third replacement layer, 1B/7B scaling, quantization, or a standalone Mamba model.

The primary endpoint is fresh token-weighted next-token loss. Lower local discrepancy, lower entropy, coherent text, or a development CE improvement is not success.

## 2. Frozen Hybrid and Common Training

Attention layers 0 and 1 are wrapped sequentially with fresh `MambaMixer` modules of state sizes 64 and 96. The conditional rank-16 residual transport is

> \(y=(1-\alpha)A(h)+\alpha T_q(h,M(h))\),

> \(T_q(h,m)=m+q\odot W_{up}\operatorname{SiLU}(W_{down}[\operatorname{RMS}(h);\operatorname{RMS}(m);\operatorname{RMS}(h)-\operatorname{RMS}(m)]).\)

The output projection \(W_{up}\) is exactly zero-initialized. At alpha zero, the frozen source attention output must be exactly reproduced. At the endpoint both active gates must equal one. Source embeddings, norms, MLPs, output head, unwrapped attention layers, and the teacher remain frozen. Only the fresh Mamba mixers and rank-16 transports may train.

Each condition receives 300 layer-0 and 420 layer-1 CE updates, for 720 updates total. Alpha is 0.05 for the first 10 updates of each active layer and then increases linearly to one; completed earlier layers remain at one. AdamW uses learning rate \(10^{-4}\), weight decay 0.01, and gradient-norm clipping at 1.0. One calibration sequence is used per update.

The branch-training objective is **ordinary next-token CE only** for all conditions. There is no teacher activation, output, logit, entropy, relation, gradient, or auxiliary loss. The frozen teacher is not used at inference.

## 3. Fixed Sensitivity-Basis Derivation

Before any branch training in each seed, the frozen teacher uses the first 128 eligible calibration sequences to estimate, for each active layer \(\ell\),

> \(f_{\ell,j}=\operatorname{mean}_{x,t}[(\partial\ell_{CE}(x)/\partial a_{\ell,t,j})^2]\),

where \(a_\ell\) is frozen source attention output and \(\ell_{CE}\) is teacher causal CE. The sensitivity estimate is computed once, detached, and receives no optimizer gradient.

The correct fixed gain is exactly

> \(q_{\ell,j}=\operatorname{RenormMean1}[\operatorname{clip}(\sqrt{f_{\ell,j}/(\operatorname{mean}(f_\ell)+10^{-12})},0.50,2.00)].\)

The topology-destroyed gain is exactly \(q^{perm}=\operatorname{roll}(q,\lfloor d/2\rfloor)\). No formula, exponent, bound, rank, state size, derivation sequence count, cyclic shift, or other hyperparameter can change after seed 20260911 starts. The basis calculation is shared pre-training bookkeeping, not an optimizer update and not an extra CE update.

## 4. Conditions and Causal Controls

| Condition | Fixed correction gain | Teacher-derived optimizer objective | Purpose |
|---|---|---|---|
| **A — Unit conditional CE transport** | Unit vector | None | Ordinary CE-only transport baseline. |
| **B — Correct sensitivity-basis CE transport** | Correctly ordered \(q\) | None | Tests correct downstream-sensitivity capacity allocation. |
| **C — Permuted sensitivity-basis CE transport** | Same values, cyclically shifted | None | Preserves gain spectrum and all compute while breaking residual-coordinate correspondence. |

All three conditions share within-seed fresh Mamba and transport initialization, parameter count, rank, state sizes, source backbone, optimizer, CE objective, alpha schedule, update count, calibration order, development diagnostics, prompts, and final evaluator. The teacher never contributes an optimizer loss and is not used at inference.

## 5. Data Isolation

| Role | WikiText-2 raw split | Eligible sequences | Permitted use |
|---|---|---:|---|
| Sensitivity derivation | Train | First 128 | Frozen teacher gradients before branch training only. |
| Calibration | Train | First 1,024 | CE optimization only. |
| Development | Validation | 65–128 | Fixed diagnostics only. |
| Fresh final evaluation | Test | 641–768 | Loaded, tokenized, and scored only after all three conditions finish all 720 updates in a seed. |

Experiments 014–015 used test sequences 1–128; Experiment 017 used 129–256; Experiment 018 used 257–384; Experiment 019 used 385–512; and Experiment 020B used 513–640. Experiment 021B must skip the first 640 eligible test sequences. No code path may request the test split before every condition completes training in the active seed.

The five paired seeds are `20260911`, `20260912`, `20260913`, `20260914`, and `20260915`.

## 6. Measurements

| Metric | Partition | Role |
|---|---|---|
| Token-weighted next-token loss and perplexity | Fresh final slice | **Primary endpoint** |
| Teacher-relative fresh-loss gap | Fresh final slice | Scale-readiness safeguard |
| Token entropy and 20-bin top-token ECE | Fresh final slice | Diagnostics only |
| Sensitivity-weighted and unweighted attention-output discrepancy | Development | Mechanism diagnostics only |
| Post-layer-1 and post-layer-2 residual drift | Development | Amplification diagnostics only |
| Correction RMS ratio, effective rank, and gradient allocation | Development/training traces | Capacity-use diagnostics only |
| Fixed greedy prompts | Alpha-one final endpoint | Coherence sanity check only |

After all conditions train, each branch is evaluated at alpha zero on the fresh slice to verify exact source reconstruction, then at alpha one for the primary endpoint.

## 7. Acceptance Criteria

| Category | Rule |
|---|---|
| Integrity | Every condition reproduces frozen teacher fresh loss at alpha zero within \(10^{-5}\) in all five seeds. |
| Completion | Both gates equal one and every final loss is finite in all conditions and seeds. |
| Isolation | Every result records fresh test loaded after all conditions completed training. |
| Correct basis versus unit CE | B beats A in at least 4/5 paired seeds and mean \(A-B\geq0.03\) nats/token. |
| Correct basis versus permuted control | B beats C in at least 4/5 paired seeds and mean \(C-B\geq0.03\) nats/token. |
| Interpretation control | C must not match or beat B at the endpoint; otherwise correct residual-coordinate sensitivity is not causally established. |
| Scale readiness | B remains within 0.15 nats/token of teacher, B ECE exceeds A by no more than 0.01 in every seed, and continuations are coherent in at least 4/5 seeds. |
| Authorization | All preceding criteria pass before a third replacement layer, 1B/7B scaling, or quantization may be considered. |

A smaller development sensitivity discrepancy cannot substitute for either matched fresh-loss comparison. A B-versus-A effect without a B-versus-C effect is not evidence that correct teacher sensitivity coordinates transferred. A C advantage rejects the stated sensitivity-correspondence mechanism for this protocol.

## 8. Reporting Boundary

The final report must distinguish a frozen pretrained source backbone from fresh Mamba branches. It must not call the hybrid a converted standalone Mamba, claim pretraining is unnecessary, or infer success from prompt fluency, calibration, entropy, or any local diagnostic unless the complete paired fresh-loss rule passes.

Runtime repairs are permitted only if they preserve all declared mathematics, split isolation, budgets, seeds, controls, and acceptance criteria; they must be recorded before rerunning seed 20260911.
