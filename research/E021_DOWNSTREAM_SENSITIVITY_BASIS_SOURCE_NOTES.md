# Experiment 021 Source Notes: Downstream-Sensitivity Basis for Residual Transport

**Status:** Hypothesis-development notes only. These notes do not authorize implementation, test access, scaling, or a success claim.

## Evidence Boundary from Completed Paradom Studies

Experiments 014, 017, 019, and 020B each demonstrate the same critical dissociation: a teacher-derived local diagnostic can improve substantially without producing a qualifying paired fresh-loss advantage over matched CE adaptation. The latest result, Experiment 020B, adds a topology-destroyed relation control: lowering post-block relation KL did not establish that the correct teacher token topology mattered at the final language-model endpoint.

Experiment 018 remains the most useful architectural signal. An input-conditioned, deployed rank-16 residual transport map gave a small but non-qualifying CE-only improvement over an unconditional map. Its teacher-derived downstream-anchor loss added only 0.0034 nats/token. Therefore, another activation, relation, entropy, or downstream-output auxiliary would repeat a class already strongly constrained by the evidence.

> The remaining question is not whether a local teacher target can be made smaller. It is whether a **frozen downstream sensitivity structure** can be used to parameterize a residual interface so that ordinary end-to-end CE allocates limited correction capacity to globally consequential residual directions.

## Literature Findings

Srinivas and Fleuret describe Jacobian matching as matching sensitivity of outputs to inputs rather than only activation values. Their abstract establishes an equivalence between Jacobian matching and noisy-input distillation and reports empirical transfer-learning benefits in their image-domain setting [1]. This motivates sensitivity as a legitimate hypothesis class, but it does **not** establish a benefit for autoregressive language models, Mamba/Transformer hybrids, residual-stream interface replacement, or the present limited-data protocol.

Liu et al. explicitly argue that direct output or intermediate-feature imitation may be unsuitable across heterogeneous architectures, and instead use projected alignment spaces [2]. This supports avoiding a claim that Transformer attention and Mamba coordinates should be directly identified. Here, however, the proposed sensitivity vector lives in the **frozen downstream residual-stream coordinate system**—the system that consumes the replacement output—not in Mamba’s internal coordinate system. That is a narrower claim that requires direct testing.

Network-morphism work formalizes function-preserving changes as a useful starting point for architecture changes [3]. The existing alpha-zero gate and zero-initialized transport correction already serve this integrity role. A downstream-sensitivity basis would not be an exact architecture morphism; it must retain alpha-zero tests and matched controls.

## Candidate Structural Mechanism

At a replacement layer, let \(a\) be the frozen source attention output supplied to the residual stream and let \(\ell_{CE}\) be the frozen teacher’s causal-language-model cross-entropy on calibration text. Compute an empirical diagonal downstream sensitivity vector only once, before branch training:

> \(f_j=\operatorname{mean}_{x,t}\left[(\partial\ell_{CE}(x)/\partial a_{t,j})^2\right]\).

Convert it into a bounded, mean-one output basis gain \(q\), for example

> \(q_j=\operatorname{clip}\left((f_j/(\operatorname{mean}(f)+\epsilon))^{\gamma/2},q_{min},q_{max}\right)\), then renormalize \(\operatorname{mean}(q)=1\).

The deployed conditional transport remains rank 16 and is trained **only with next-token CE**, but its correction enters the frozen downstream residual coordinates through this fixed basis:

> \(T_{q}(h,m)=m+q\odot W_{up}\operatorname{SiLU}(W_{down}[\operatorname{RMS}(h);\operatorname{RMS}(m);\operatorname{RMS}(h)-\operatorname{RMS}(m)]).\)

The teacher supplies no activation target, logit target, relation target, output-sharpening term, gradient target, or inference signal. It supplies only a fixed pre-training coordinate allocation for the adapter’s limited correction capacity. The primary training objective remains end-to-end CE through the frozen hybrid.

## Essential Causal Controls

Any endpoint experiment must compare three equal-rank, equal-parameter, equal-budget CE-only conditions:

| Condition | Fixed correction basis | Teacher-derived optimizer objective | Interpretation |
|---|---|---|---|
| Plain conditional transport | Unit gain | None | CE-only architecture baseline. |
| Correct sensitivity basis | Correctly ordered \(q\) | None | Tests whether teacher downstream sensitivity is useful as a structural prior. |
| Permuted sensitivity basis | A fixed cyclic permutation of the same \(q\) values | None | Preserves basis spectrum and compute while breaking correspondence to residual channels. |

A benefit of the correct basis over plain CE is insufficient. It must also beat the permuted-basis control on a fresh endpoint. Otherwise, any difference can be attributed to generic gain reparameterization or chance rather than the teacher’s correct downstream coordinate sensitivity.

## Recommended Staging

Because gain exponent \(\gamma\) and bounds are hyperparameters, a development-only, no-test-access pilot should select them before an endpoint experiment. The pilot must use a newly recorded development coefficient rule and cannot compare or inspect any final test output. If no correct basis is both CE-safe and better than the permuted control on the development diagnostic rule, the final experiment is cancelled.

Neither the cited literature nor an improved development diagnostic can establish knowledge transfer. Only a separate, preregistered multi-seed fresh-loss result that beats both matched controls can do so.

## References

[1] [Srinivas, S., and Fleuret, F. “Knowledge Transfer with Jacobian Matching.” ICML 2018.](https://proceedings.mlr.press/v80/srinivas18a.html)

[2] [Liu, Y. et al. “Cross-Architecture Knowledge Distillation.” ACCV 2022.](https://openaccess.thecvf.com/content/ACCV2022/html/Liu_Cross-Architecture_Knowledge_Distillation_ACCV_2022_paper.html)

[3] [Wei, T. et al. “Network Morphism.” ICML 2016.](https://proceedings.mlr.press/v48/wei16.html)
