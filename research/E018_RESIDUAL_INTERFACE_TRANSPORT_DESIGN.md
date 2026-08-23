# Experiment 018 Design: End-to-End Residual Interface Transport

**Status:** Architecture-design proposal only. No model has been trained under this design, and no claim of successful transfer is made.

## 1. Why a New Design Is Needed

Experiments 014–017 established a consistent dissociation: teacher-guided losses improved the local quantity they optimized, while equal-capacity CE-only adaptation produced the better held-out two-layer hybrid endpoint. The failed quantities include attention-output values, source-logit fitting, directional responses, and featurewise output moments. The next proposal must therefore avoid treating raw Mamba-to-attention similarity as its primary target.

Cross-architecture distillation literature similarly warns that directly matching heterogeneous intermediate representations can be ineffective because their architecture-specific content diverges. It instead motivates learned aligned spaces or projectors [1] [2]. A language-model intermediate-layer study reports that its LoRA alignment adapter did not beat baseline distillation and highlights the risk of an alignment map used during training but absent at inference [3]. The key design rule is therefore:

> **The deployed Mamba branch must contain the learned interface map, and next-token CE through the frozen downstream backbone must remain the dominant training objective.**

The proposal below is deliberately not a function-preserving Transformer-to-Mamba conversion. Exact function preservation requires a construction that makes the transformed student equal the teacher for all inputs, a condition known from network-morphism work [4]. No such attention-to-Mamba construction has been derived here. Instead, identity initialization preserves the existing interface component of the *new* transport map; alpha gating separately preserves the original Transformer at alpha zero.

## 2. Mechanism: End-to-End Residual Interface Transport

Let `h` be the frozen attention-input representation at a replaced layer and `m = M_phi(h)` the fresh Mamba output. The deployed replacement output is

> `M_transport(h) = m + W_up SiLU(W_down q(h, m))`

where `W_down` has rank `r=16`, `W_up` maps the bottleneck back to the hidden dimension, and

> `q(h, m) = [RMS(h), RMS(m), RMS(h) − RMS(m)]`.

`RMS(·)` is a parameter-free per-token root-mean-square normalization. `W_down` is standard-initialized and `W_up` is zero-initialized, so the transport correction begins at exactly zero. The final gated wrapper remains

> `y = (1 − alpha) A(h) + alpha M_transport(h)`.

At `alpha=0`, the frozen attention `A` alone supplies the output; this must reproduce the source model exactly. At `alpha=1`, no attention output contributes to `y`. The transport map uses only `h` and `m`, which are available in the deployed hybrid; it does not require teacher activations, teacher attention outputs, or a teacher model at inference.

| Component | Purpose | Present at inference? | Initialization |
|---|---|---:|---|
| Fresh Mamba `M_phi` | Alternative sequence operator | Yes | Fresh random Mamba initialization |
| Conditional transport `W_up SiLU(W_down q(h,m))` | Correct a Mamba output using input–output residual context | Yes | Zero output correction |
| Gate alpha | Safe staged substitution from frozen attention to transport branch | Yes | Starts at zero; ends at one |
| Source teacher computation | Calibration-only target/diagnostic source for one condition | No | Frozen |

This differs from the rejected diagonal calibrator: it can mix hidden features through a nonlinear low-rank map and condition the correction on both the input residual state and Mamba output. It also differs from direct local fitting: its default objective is end-to-end CE after the frozen downstream computation.

## 3. Three Matched Conditions

All conditions use an equal-rank transport module, the same Mamba parameterization, the same initialization within each paired seed, the same calibration data, the same sequential layer order, the same optimizer update budget, and the same alpha schedule. CE is primary in every condition.

| Condition | Transport input `q` | Objective | Causal question |
|---|---|---|---|
| **A — Unconditional CE transport** | `[RMS(m), RMS(m), RMS(m)]` | CE only | Does added transport capacity without source-residual conditioning help? |
| **B — Conditional CE transport** | `[RMS(h), RMS(m), RMS(h) − RMS(m)]` | CE only | Does the new deployed conditional interface improve the global endpoint beyond capacity-matched transport? |
| **C — Conditional downstream-anchor transport** | Same as B | `CE + lambda × L_anchor` | Does a small teacher-derived *downstream* auxiliary add value beyond the same conditional transport? |

The unconditional control has exactly the same input dimension and transport parameter count as B and C. Repeating `RMS(m)` three times keeps all transport weights active while forbidding direct dependence on `h`. This isolates **conditional residual interface transport**, rather than raw parameter count.

## 4. Teacher Auxiliary: Downstream Anchor, Not Branch Imitation

Condition C receives a single small teacher-derived calibration auxiliary. For replacement layer `l`, let `F_l` denote the frozen continuation from that layer’s attention-output site through the remaining residual and MLP computation of the same Transformer block. The source reference is

> `z_teacher = F_l(A(h))`,

and the hybrid representation is

> `z_hybrid = F_l(M_transport(h))`.

The auxiliary is computed after parameter-free RMS normalization:

> `L_anchor = mean_{batch,token}(1 − cosine(RMS(z_hybrid), RMS(z_teacher))) + 0.25 × relative_MSE(RMS(z_hybrid), RMS(z_teacher))`.

The primary loss remains `L = CE + lambda L_anchor` with fixed `lambda = 0.05`. There is **no raw attention-output value loss, moment loss, tangent loss, or teacher-logit KL loss** in Experiment 018. The anchor is selected because Experiment 016 located amplification after the intervention, while Experiments 014–017 demonstrate that matching pre-amplification branch statistics is insufficient.

Condition C is not assumed to be superior. A result where B beats C would show that this teacher auxiliary is also harmful or redundant. A result where C beats B would be the first evidence in this project that a teacher-derived signal improves an end-to-end interface beyond a matched CE-only transport architecture.

## 5. Training Schedule and Safeguards

Replacement remains sequential: layer 0, then layer 1. At each active layer, every condition receives an identical end-to-end schedule: alpha is 0.05 during the initial warm-up updates and rises linearly to 1.0 over the remainder. Earlier separate local-fitting phases are intentionally removed, because they made local matching the dominant optimization target.

| Safeguard | Requirement |
|---|---|
| Source integrity | Every condition must reproduce source final-slice loss at alpha zero within `1e−5` |
| Deployability | The transport map may not consume teacher values at inference |
| Capacity control | All conditions use the same Mamba, rank-16 transport dimensions, parameter count, and initialization |
| Optimization control | Same CE data, sequence order, update count, learning rate schedule, alpha schedule, and gradient clipping |
| Teacher isolation | Only C gets `L_anchor`; A and B never compute teacher activations in their optimizer objective |
| Evaluation isolation | The fresh final test slice is requested only after every condition has completed all optimization in a seed |
| Endpoint discipline | No alpha below one may be presented as a complete replacement result |

## 6. What Would Count as Evidence

The design separates three claims.

| Comparison | Minimum prespecified evidence | Interpretation if passed |
|---|---|---|
| B versus A | B lower final fresh-test loss in at least 4/5 paired seeds and mean `A − B >= 0.05` | The input-conditioned deployed interface is more useful than capacity-matched unconditional transport under CE |
| C versus B | C lower final fresh-test loss in at least 4/5 paired seeds and mean `B − C >= 0.03` | The downstream teacher anchor adds value beyond the same CE-optimized conditional interface |
| C versus A and frozen teacher | C meets both comparisons and is within 0.15 nats/token of teacher | A limited small-model cross-architecture transfer result is supported |

Failing any comparison forbids a third-layer replacement and forbids scaling. B beating A but C failing to beat B would support an interface-architecture effect, not teacher-specific knowledge transfer. C beating B but not meeting teacher proximity would support a tentative auxiliary benefit but not scale-up.

## 7. Risks and Falsifiers

The proposal is deliberately high risk. The transport map may simply provide more convenient end-to-end capacity, in which case A and B may be tied. Conditioning on `h` may harm rather than help. The downstream anchor may create another harmful local proxy even though it is measured later than prior targets. The conditions and rules above make all of these outcomes informative.

This design also does not prove that a standalone Mamba model has inherited a Transformer’s behavior. Even a positive outcome would concern a **two-layer hybrid inside a frozen Transformer checkpoint**. It would only justify the next small-scale replacement study after independent replication.

## References

[1] [Liu et al. *Cross-Architecture Knowledge Distillation* (ACCV 2022).](https://openaccess.thecvf.com/content/ACCV2022/html/Liu_Cross-Architecture_Knowledge_Distillation_ACCV_2022_paper.html)

[2] [Wang et al. *One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation* (NeurIPS 2023).](https://neurips.cc/virtual/2023/poster/72626)

[3] [Kuratsu et al. *Aligning Sizes of Intermediate Layers by LoRA Adapter for Knowledge Distillation* (2025).](https://aclanthology.org/2025.insights-1.10/)

[4] [Jaderberg. *Towards a More Complete Theory of Function Preserving Transforms* (2024 version).](https://arxiv.org/abs/2410.11038)

[5] [Nagrani et al. *Knowledge Distillation for Large Language Models Through Residual Learning* (2026).](https://www.amazon.science/publications/knowledge-distillation-for-large-language-models-through-residual-learning)
