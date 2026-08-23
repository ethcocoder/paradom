# Pre-Experiment-018 Evidence Synthesis: What the Completed Transformer-to-Mamba Studies Establish

**Scope:** This document consolidates the completed small-model causal studies before any new architecture or objective is proposed. It does not introduce a success claim, alter prior protocols, or authorize scaling.

## Question Under Review

The broad research ambition is to determine whether useful behavior from a trained Transformer checkpoint can be transferred into a different sequence architecture with less data and compute than full pretraining. The decisive experimental question is narrower:

> Does a teacher-specific cross-architecture mechanism improve a Mamba replacement beyond an equal-capacity, equal-budget Mamba module trained directly on next-token cross-entropy in the same frozen Transformer backbone?

The answer remains **no for every teacher-guided mechanism tested through Experiment 017**. This does not show that all transfer is impossible; it specifies the evidence threshold that remains unmet.

## Causal Evidence Matrix

| Study | Tested intervention | Local outcome | Complete held-out endpoint | Result relative to equal-capacity CE control |
|---|---|---|---|---|
| Experiment 013 | Functional attention-trajectory fitting versus direct static tensor mapping | Functional fitting was operationally stable | Two-layer hybrid retained English-like behavior | Functional fitting beat static mapping, but CE-only was not tested in this study |
| Experiment 014 | Attention-value and teacher-logit functional objective | Layer-1 value NMSE: 0.1489 versus 1.1650 for CE-only | Functional: 4.1345; CE-only: 4.0923 | **CE-only won 5/5** |
| Experiment 015 | Directional matching plus bounded adaptive gate | Directional NMSE: 0.9912 versus 1.0699 for value-only | Adaptive branch stalled at layer-1 alpha 0.80 in 5/5 | No valid adaptive endpoint; CE-only beat completed value-functional endpoint 5/5 |
| Experiment 016 | Development-only alpha boundary measurement | Smooth degradation, static branch mismatch, downstream amplification | Not an endpoint experiment; test split prohibited | Identified a diagnostic signature, not a transfer mechanism |
| Experiment 017 | Featurewise output mean/variance calibration with equal parameter count | Layer-1 log-variance mismatch: 0.1877 versus 1.2757 for value-functional | Moment: 4.4173; value: 4.0430; CE: 4.0089 | **CE-only won 5/5** against both teacher-guided conditions |

## What Is Reliably Supported

The following statements are supported by the accumulated experiments.

| Statement | Evidence | Boundary of the claim |
|---|---|---|
| A frozen Transformer checkpoint can host two fully active fresh Mamba replacement modules while preserving coherent English behavior after limited CE adaptation | Experiments 014, 015 completed controls, and 017 | This is a hybrid adaptation result, not a pure-Mamba conversion |
| Direct static Transformer-to-Mamba tensor projection is not an adequate transfer mechanism in this setting | Experiments 005–008 and 013 | It does not rule out all learned weight-space transforms |
| Teacher-guided local objectives can strongly improve their chosen branch diagnostics | Value, directional, and moment objectives each improved their corresponding local metric | The local metrics have repeatedly failed as predictors of held-out language loss |
| The layer-1 gate boundary is a smooth interface degradation rather than a numerical cliff | Experiment 016 development-only alpha sweep | The measurement does not identify a single frozen-block operation that causes the degradation |
| Downstream frozen computation amplifies the injected branch discrepancy | Experiment 016 post-layer-2 drift exceeded post-layer-1 drift | This is a signature, not a proof of a complete causal pathway |

## What Has Been Rejected

| Claim or mechanism | Status | Basis |
|---|---|---|
| Static weight mapping preserves useful language behavior across Transformer and Mamba architectures | Rejected in the tested setting | Converted models were worse than random branches |
| Attention value plus teacher-logit functional distillation beats matched CE adaptation | Rejected | CE-only won all five Experiment 014 paired seeds |
| Better local directional response matching repairs the end-to-end endpoint | Rejected for the tested objective/controller | Adaptive condition stalled at alpha 0.80 in all five Experiment 015 seeds |
| Featurewise output mean/variance matching repairs the end-to-end endpoint | Rejected for the tested diagonal calibrator and loss | Moment condition lost all five Experiment 017 seeds on a fresh test slice |
| Local diagnostic improvement establishes a successful knowledge-transfer mechanism | Rejected | Every teacher-guided objective displayed this dissociation |
| Training from scratch is generally unnecessary | Not established | No teacher-guided condition yet outperforms matched CE-only adaptation |

## The Unresolved Causal Bottleneck

The repeated failure is not merely that Mamba cannot approximate attention locally. Mamba can be trained to reproduce values, selected directional behavior, or featurewise moments better than CE-only branches. The bottleneck is instead:

> **A locally fitted Mamba output is not a globally compatible residual-stream intervention for the frozen downstream Transformer.**

The key distinction is between an **attention-output approximation problem** and an **interface-transport problem**. Prior teacher-guided losses attempted to make Mamba resemble attention at selected local measurements. The frozen downstream blocks, however, process the entire residual-stream trajectory through nonlinear normalization, MLP, and subsequent attention computations. A successful intervention must preserve behavior that matters after this downstream computation, not merely the output statistics measured immediately at the replacement site.

## Design Constraints for a Genuinely New Hypothesis

Any next mechanism must satisfy all of the following conditions. Otherwise it repeats a ruled-out class of local proxy studies.

| Constraint | Rationale |
|---|---|
| The trainable interface must be optimized using an **end-to-end frozen-backbone language objective** | CE-only is the only tested signal that consistently predicts the global endpoint |
| Teacher activations, logits, and layer measurements must be diagnostic or auxiliary, not the dominant target by default | Dominant local targets repeatedly diverted capacity away from held-out language behavior |
| Every condition must have equal trainable parameter count and identical initialization within each paired seed | Capacity must not confound any apparent teacher benefit |
| The intervention must be identity-initialized and preserve the alpha-zero teacher endpoint exactly | Source preservation remains a non-negotiable integrity check |
| The architecture must be structurally different from pointwise/diagonal local fitting | Values, directional responses, and featurewise moments have already been tested and rejected as primary mechanisms |
| CE-only, end-to-end transport-only, and transport-plus-teacher-auxiliary controls are required | These distinguish ordinary adaptation, a new architecture effect, and any incremental teacher-specific signal |
| A fresh final test slice must remain inaccessible until all conditions train | Prevents selection on evaluation outcomes |

## Candidate Research Direction

The candidate for literature review and later protocol design is an **end-to-end residual interface transport map** around the Mamba branch. Conceptually,

> `h_out = h_in + T_theta(M(h_in), h_in)`

where `T_theta` is a constrained, identity-initialized, residual map. Unlike the prior diagonal featurewise calibrator, this map can use the branch output and the current residual input jointly. Its primary loss would be next-token CE through the frozen downstream backbone. Teacher signals would remain diagnostics or carefully ablated auxiliaries.

This is not yet an experimental proposal. It is an architectural hypothesis that must be compared with a parameter-matched CE-only baseline and an equivalent transport map without teacher-derived loss. The next phase will review whether known methods in representation alignment, residual adapters, end-to-end knowledge distillation, and function-preserving network transformation provide sound mechanisms and controls for such a study.

## Current Research Decision

No 1B scale-up, 4-bit quantization intervention, third replacement layer, or pure-Mamba conversion is warranted at this point. The next work is a constrained design study, not an attempt to rescue a negative local objective by adding scale or compression.
