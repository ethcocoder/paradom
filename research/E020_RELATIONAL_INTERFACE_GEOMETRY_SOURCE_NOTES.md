# Experiment 020 Candidate: Relational Interface Geometry — Source Notes

**Status:** Literature-and-evidence review only. This document neither changes any completed experiment nor authorizes training, final-test access, scaling, a third replacement layer, or quantization.

## Motivation From Completed Paradom Evidence

The relevant internal finding is not that fresh Mamba branches cannot reproduce selected Transformer quantities. They can improve local values, directional behavior, and moments. The repeated failure is that these local improvements have not become a globally compatible residual-stream intervention for the frozen downstream Transformer. Experiment 016 measured smooth alpha-dependent degradation, a stable branch mismatch with featurewise geometry error, and larger post-layer-2 than post-layer-1 drift. Experiment 018 then made the transport intervention conditional on the current residual input and optimized it end-to-end through CE. That architecture was directionally better than unconditional transport but did not reach its threshold; a direct post-intervention downstream anchor added only 0.0034 nats/token. Experiment 019 likewise found a small but non-qualifying teacher-temperature effect after the shared transport stage.

The surviving hypothesis must therefore avoid a return to pointwise attention-value, directional, or moment matching. A candidate auxiliary should be measured **after** the deployed intervention has propagated through frozen computation and should not depend on the coordinate-by-coordinate equality of Mamba and attention outputs.

## Relevant External Findings

| Source | Narrow finding | Relevance and limitation for Paradom |
|---|---|---|
| Park et al., *Relational Knowledge Distillation* (CVPR 2019) [1] | The paper frames relational KD as transferring mutual relations among examples, and instantiates distance- and angle-wise losses rather than only individual output activations. | This motivates comparing structure that is invariant to many coordinate choices. It is not evidence for Mamba, language modeling, or frozen-hybrid deployment. |
| Kim et al., *Relational Representation Distillation* (2024) [2] | The authors describe relation-based methods as preserving pairwise similarities and formulate relational consistency by matching distributions of normalized similarities. | This supports a row-normalized token-relation loss rather than raw hidden-state regression. Their experiments are vision-oriented and use memory buffers, neither of which should be assumed necessary or effective here. |
| Liu et al., *Cross-Architecture Knowledge Distillation* (ACCV 2022) [3] | The paper argues that direct mimicry of intermediate features can be unsuitable across architecture families and uses projected alignment spaces. | This agrees with Paradom’s direct-matching failures. It motivates a geometry-based target, but does not validate the exact target or objective coefficient. |
| Hao et al., *One-for-All* (NeurIPS 2023) [4] | The work reports heterogeneous feature divergence and projects intermediate features into a more aligned latent space to reduce architecture-specific information. | This reinforces the need to discard coordinate-specific details. It does not establish that logits or a generic projection are the correct space for this frozen Transformer–Mamba hybrid. |

> **Boundary:** These sources offer a hypothesis-generation analogy. None proves that relational loss transfers knowledge from Transformer attention to Mamba, and none relaxes the requirement to beat a paired CE-only endpoint on an untouched language-model test slice.

## Candidate Mechanism: Post-Block Token-Relation Geometry

For each sequence, let \(H_T\in\mathbb{R}^{L\times d}\) be a frozen-teacher residual stream and \(H_S\in\mathbb{R}^{L\times d}\) the hybrid residual stream captured at the **same downstream frozen block input**, after both active Mamba transport substitutions have contributed. Normalize each token vector, form token-to-token cosine-similarity matrices, and convert each row to a distribution:

> \(R(H)=\operatorname{softmax}(\operatorname{RMS}(H)\operatorname{RMS}(H)^\top/\tau_r)\).

The candidate teacher auxiliary is

> \(L_{rel}=\operatorname{mean}_{i}\operatorname{KL}(R(H_T)_i\parallel R(H_S)_i)\).

This objective has three intended properties. First, it is **post-deployment**: it is measured after the replacement has propagated to a common frozen downstream location. Second, it is **relational**: it compares sequence-position geometry rather than raw coordinate equality, tokenwise feature moments, or an individual attention output. Third, it leaves end-to-end token CE as the primary optimizer signal and final primary endpoint.

## Required Controls for a Future Experiment

A valid endpoint study must keep the same two fresh Mamba mixers, rank-16 conditional transport, alpha schedule, trainable parameter count, initialization, optimizer, calibration order, and total update budget in every condition. The candidate comparison is:

| Condition | Objective | Role |
|---|---|---|
| **A — Conditional CE transport** | \(CE\) | Ordinary end-to-end adaptation baseline. |
| **B — Teacher relational geometry** | \(CE+\lambda L_{rel}\) | Tests whether correctly indexed teacher relation geometry adds a material benefit. |
| **C — Permuted-relation control** | \(CE+\lambda L_{perm}\) | Uses the same teacher relation distributions, coefficient, and computation but applies a fixed non-identity token-position permutation to the target columns, destroying the original relational correspondence. |

The control is not expected to be helpful. Its purpose is to test whether any observed gain in B depends on the **correct teacher relational topology**, rather than the mere presence of an extra loss, similar compute, or distributional concentration. The teacher must not be used at inference in any condition.

## Risks and Mandatory Safeguards

The candidate has serious falsification risks. Pairwise losses may constrain a sequence too rigidly, may dominate CE, and could simply produce another local diagnostic that fails globally. Short sequences also make relation matrices inexpensive but potentially noisy. These risks demand a development-only pilot to select a single numerically stable coefficient before a locked endpoint study. The pilot may load only WikiText-2 train and validation partitions; it must not request or inspect the test partition.

If a future endpoint protocol is authorized, it must use a new untouched final test range after Experiment 019’s eligible test sequences 385–512, retain alpha-zero exactness and alpha-one checks, and require B to beat **both** A and C by preregistered mean loss margins and paired win counts. A reduced relational diagnostic, lower entropy, or plausible prompts cannot substitute for that criterion.

## References

[1] [Park et al. *Relational Knowledge Distillation.* CVPR 2019.](https://arxiv.org/abs/1904.05068)

[2] [Kim et al. *Relational Representation Distillation.* 2024.](https://arxiv.org/html/2407.12073v3)

[3] [Liu et al. *Cross-Architecture Knowledge Distillation.* ACCV 2022.](https://openaccess.thecvf.com/content/ACCV2022/html/Liu_Cross-Architecture_Knowledge_Distillation_ACCV_2022_paper.html)

[4] [Hao et al. *One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation.* NeurIPS 2023.](https://neurips.cc/virtual/2023/poster/72626)
