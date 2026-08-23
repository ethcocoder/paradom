# Experiment 018: End-to-End Residual Interface Transport

**Status:** Pre-registered protocol. Implementation and seed execution are prohibited until this document is committed unchanged.

## 1. Objective

Experiment 018 tests whether a deployed, input-conditioned residual transport map around a fresh Mamba replacement can improve a two-layer frozen-Transformer hybrid beyond an equal-parameter unconditional transport map trained by next-token cross-entropy. It also tests whether a small **downstream teacher anchor** adds a teacher-specific advantage beyond the identical CE-trained conditional transport.

This is not a test of direct attention imitation, static weight mapping, output moments, or standalone Mamba conversion. It is a new architecture-and-objective test motivated by the repeated finding that local Mamba-to-attention matching metrics did not predict global language-model behavior.

## 2. Frozen Source and Replacement Endpoint

The frozen source checkpoint is `HuggingFaceTB/SmolLM-135M`. Attention modules at layers 0 and 1 are sequentially wrapped and replaced by fresh Mamba mixers with state sizes 64 and 96, respectively. Embeddings, all layer norms, all MLPs, the output head, and attention modules other than the two wrapped sites remain frozen.

At an active replacement layer, the wrapper is

> `y = (1 - alpha) A(h) + alpha T_theta(h, M_phi(h))`,

where `A` is the original frozen attention module, `M_phi` is a fresh Mamba mixer, and `T_theta` is the deployed transport function. Both replacement gates must end at `alpha=1.0`; a partial-gate model has no valid final score.

## 3. Deployed Transport Architecture

Let `m = M_phi(h)` and let `RMS(x) = x / sqrt(mean_d(x_d^2) + epsilon)` tokenwise over hidden dimension `d`. Let

> `q_cond(h,m) = concat(RMS(h), RMS(m), RMS(h) - RMS(m))`

and

> `q_uncond(h,m) = concat(RMS(m), RMS(m), RMS(m))`.

For all conditions, the transport output is

> `T_theta(h,m;q) = m + W_up SiLU(W_down q)`.

`W_down` maps `3 × hidden_size` into rank 16; `W_up` maps rank 16 back to the hidden size. `W_down` is standard initialized and `W_up` is exactly zero initialized. Thus, before training, `T_theta = m` exactly. The transport module is used at training and inference in every condition.

## 4. Matched Conditions

| Condition | Transport input | Primary loss | Teacher-derived optimizer signal | Causal role |
|---|---|---|---|---|
| A — Unconditional CE transport | `q_uncond` | Token CE | None | Parameter-matched transport-capacity control |
| B — Conditional CE transport | `q_cond` | Token CE | None | Tests deployed residual-state conditioning |
| C — Conditional downstream-anchor transport | `q_cond` | `CE + 0.05 × L_anchor` | Downstream anchor only | Tests whether a teacher-derived post-intervention target adds value beyond B |

All conditions have identical fresh Mamba mixers, rank-16 transport modules, parameter count, paired random initialization, trainable-parameter set, optimizer, data order, update count, layer order, alpha schedule, and fixed prompts. The only differences are the transport input in A versus B/C and the explicitly declared auxiliary in C.

## 5. Downstream Teacher Anchor

For condition C only, `L_anchor` is computed on calibration examples and only during active-layer training. Let `h` be the layer input. The frozen source block gives

> `r_A = h + A(norm_in(h))`
>
> `z_A = r_A + MLP(norm_post(r_A))`.

The hybrid branch gives

> `r_T = h + T_theta(h, M_phi(norm_in(h)))`
>
> `z_T = r_T + MLP(norm_post(r_T))`.

All norm and MLP modules in these expressions are the frozen source components. The auxiliary is

> `L_anchor = mean(1 - cosine(RMS(z_T), RMS(z_A))) + 0.25 × mean((RMS(z_T) - RMS(z_A))²)`.

No condition uses a raw attention-output value loss, featurewise moment loss, directional finite-difference loss, source-logit KL loss, or static-weight projection. The only teacher-derived optimizer signal is this post-intervention anchor in condition C.

## 6. Data, Budget, and Isolation

| Role | WikiText-2 raw split | Quantity | Use |
|---|---|---:|---|
| Calibration | Train | First 1,024 eligible sequences | Optimization only |
| Development | Validation | First 64 eligible sequences | Diagnostics only; no objective, model, or schedule selection after seed 1 begins |
| Fresh final evaluation | Test | Eligible sequences 257–384, 128 sequences | Requested only after all three conditions complete all updates in a seed |

Experiments 014–015 used eligible test sequences 1–128 and Experiment 017 used sequences 129–256. Experiment 018 skips the first 256 eligible test sequences. The final slice must not be loaded, tokenized, inspected, or scored until all three conditions have completed every optimization update for the active seed.

The paired seeds are `20260861`, `20260862`, `20260863`, `20260864`, and `20260865`. Each condition receives 720 optimizer updates: 300 updates while replacing layer 0, then 420 updates while replacing layer 1. The optimizer is AdamW with learning rate `1e-4`, weight decay `0.01`, gradient-norm clipping at `1.0`, and fixed batch size one sequence. Maximum sequence length is 48 tokens. The learning rate, budget, rank, and alpha schedule may not be changed after the first seed begins.

At each layer, alpha is `0.05` for updates 1–10 and rises linearly from `0.05` to `1.0` over the remaining updates. After layer 0 reaches one, it remains at one while layer 1 trains. There is no separate local-fitting phase.

## 7. Measurements

Development diagnostics are collected without choosing among outcomes after training begins:

| Measurement | Location | Purpose |
|---|---|---|
| Token CE | Development | Global language-behavior diagnostic |
| Token-normalized source-logit KL | Development | Teacher divergence diagnostic, not an optimization target |
| Post-layer-1 and post-layer-2 residual drift | Development | Tests whether transport reduces downstream amplification signature |
| Conditional correction RMS ratio | Development | Confirms the learned transport correction is nonzero and bounded |
| `L_anchor` | Development, C only | Confirms auxiliary optimization, not success by itself |
| Fixed greedy continuations | Final endpoint only | Qualitative coherence sanity check |

The primary endpoint is token-weighted next-token loss on the fresh final test slice after both gates equal one. Each condition must also be evaluated at alpha zero on that slice after all training, solely to verify exact reproduction of the source before evaluating its alpha-one endpoint.

## 8. Prespecified Acceptance Criteria

| Category | Rule |
|---|---|
| Integrity | Every condition reproduces frozen teacher fresh-test loss at alpha zero within `1e-5` in all five seeds |
| Endpoint | Both gates reach alpha one for every condition in at least 4/5 seeds; an incomplete endpoint counts as failure |
| Stability | All valid alpha-one final losses are finite |
| Architecture comparison | B beats A on fresh test in at least 4/5 paired seeds and mean `A - B >= 0.05` nats/token |
| Teacher-specific comparison | C beats B on fresh test in at least 4/5 paired seeds and mean `B - C >= 0.03` nats/token |
| Scale readiness | C is within 0.15 nats/token of the frozen teacher and fixed continuations are coherent in at least 4/5 seeds |
| Third-layer authorization | All prior integrity, endpoint, architecture, teacher-specific, and scale-readiness criteria pass |

The test is deliberately demanding. A result where B beats A but C does not beat B supports a new architecture effect, not teacher-specific transfer. A result where C beats B but fails the remaining criteria supports only a preliminary signal. A result where A remains best rejects the current transport mechanism. None of these outcomes justify 1B scaling or quantization by themselves.

## 9. Prohibited Changes and Reporting Rules

The following are prohibited after the first seed begins: changing loss weights, rank, alpha schedule, update count, data sequence offset, test slice, prompts, optimizer, seed list, endpoint definition, or acceptance rules. A runtime failure may be fixed only if it prevents execution and the fix does not alter the declared mathematical mechanism; any such change must be documented before rerunning seed 1.

Final reporting must distinguish: (1) the ability of a Mamba-containing hybrid to perform language modeling; (2) the benefit of conditional transport; and (3) any teacher-specific benefit. No result may be described as a pure Mamba conversion, proof that pretraining is unnecessary, or evidence supporting 1B/7B scaling unless the corresponding criteria actually pass.

## 10. Status

This document authorizes only future implementation and a first-seed validation after it is committed unchanged. It does not authorize a third layer, 1B scaling, 4-bit quantization, or test-set access outside the final endpoint process.
