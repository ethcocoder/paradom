# Research Notes for the Post-Experiment-017 Interface-Transport Design

## Sources Examined

| Source | Verified finding | Relevance to the Paradom bottleneck | Constraint on interpretation |
|---|---|---|---|
| Liu et al., *Cross-Architecture Knowledge Distillation* (ACCV 2022) | Cross-architecture teacher and student features may need projection into aligned spaces rather than direct feature matching; the work uses partially cross-attention and groupwise linear projectors. | Supports treating an architecture interface as an alignment problem rather than naïve coordinate-wise regression. | Vision classification experiments do not establish that the same projectors transfer Transformer attention behavior into Mamba language-model blocks. |
| Pan et al., *Knowledge Distillation via the Target-aware Transformer* | Direct one-to-one feature regression can be suboptimal when teacher and student representations have differing semantic content; a learned correlation-based reconfiguration is proposed. | Supports avoiding another direct pointwise Mamba-to-attention loss. | The proposed reconfiguration uses teacher features as inputs to the alignment computation and is not appropriate as an inference-time hybrid interface without an explicit teacher-free deployment path. |
| Wang et al., *One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation* (NeurIPS 2023) | Heterogeneous models exhibited feature divergence under CKA; the method projects intermediate features into an aligned latent/logit-like space and attempts to filter irrelevant teacher information. | Supports the hypothesis that raw internal features retain architecture-specific components and may be harmful as dominant targets. | It is a vision KD result and does not test a frozen-backbone module replacement endpoint. |
| Kuratsu et al., *Aligning Sizes of Intermediate Layers by LoRA Adapter for Knowledge Distillation* (2025) | The paper highlights an inference mismatch when an alignment map is used only to calculate a training loss, and reports that its LoRA alignment method did not outperform baselines; its conventional intermediate-layer distillation also did not outperform vanilla KD. | Directly reinforces the Paradom finding that auxiliary intermediate alignment may not improve language-model performance. | The architecture and task differ; this is corroborating evidence, not proof of the Paradom mechanism. |
| Nagrani et al., *Knowledge Distillation for Large Language Models Through Residual Learning* (2026) | The work proposes a projector/compression stage and residual learning that lets the student improve beyond possibly imperfect teacher representations, combining projected teacher information with a supervised objective. | Suggests residual teacher information should be auxiliary to end-to-end task loss rather than an overriding local constraint. | It is student-model distillation, not Transformer-to-SSM block replacement; the project’s causal controls remain necessary. |
| Jaderberg, *Towards a More Complete Theory of Function-Preserving Transforms* | Function-preserving transforms explicitly require the transformed model to preserve the original function at initialization; residual-compatible transforms use cancellation/identity constructions. | Supports identity initialization and exact alpha-zero endpoint preservation as hard integrity constraints, not a performance mechanism by themselves. | The exact constructions target capacity changes within conventional networks and do not derive a function-preserving Transformer-attention-to-Mamba replacement. |

## Design Consequences

The literature does not provide evidence for a direct exact Transformer-to-Mamba function-preserving map. It does support three constraints for a new Paradom design:

1. **No raw coordinate-wise teacher feature loss as the primary objective.** Heterogeneous feature spaces may encode architecture-specific content, and both the literature and Experiments 014–017 show that direct local targets can underperform task-level learning.
2. **Any learned map must exist at inference.** A projector used solely in an auxiliary loss risks a train–inference mismatch. The interface map must be part of the deployed Mamba branch.
3. **Next-token CE must remain primary.** Teacher information should be incorporated as a controlled auxiliary residual signal or diagnostics, then tested against identical end-to-end transport without teacher supervision.

These are research-derived constraints, not a claim that the next design will work.

## References

[1] [Liu et al. *Cross-Architecture Knowledge Distillation* (ACCV 2022).](https://openaccess.thecvf.com/content/ACCV2022/html/Liu_Cross-Architecture_Knowledge_Distillation_ACCV_2022_paper.html)

[2] [Pan et al. *Knowledge Distillation via the Target-aware Transformer* (2022).](https://arxiv.org/abs/2205.10793)

[3] [Wang et al. *One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation* (NeurIPS 2023).](https://neurips.cc/virtual/2023/poster/72626)

[4] [Kuratsu et al. *Aligning Sizes of Intermediate Layers by LoRA Adapter for Knowledge Distillation* (2025).](https://aclanthology.org/2025.insights-1.10/)

[5] [Nagrani et al. *Knowledge Distillation for Large Language Models Through Residual Learning* (2026).](https://www.amazon.science/publications/knowledge-distillation-for-large-language-models-through-residual-learning)

[6] [Jaderberg. *Towards a More Complete Theory of Function Preserving Transforms* (2024 version).](https://arxiv.org/abs/2410.11038)
