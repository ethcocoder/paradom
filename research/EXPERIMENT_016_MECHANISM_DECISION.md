# Experiment 016 Mechanism Decision Record

## Development-Only Finding

The repeated `α=0.80 → 0.90` rejection is **not a discrete numerical cliff**. Across five reconstructed adaptive-interface runs, development CE and the controller’s source-logit KL increase smoothly as layer-1 alpha rises. The controller crosses its fixed KL threshold because a smooth, slightly accelerating curve passes through an absolute limit.

At the 0.80-to-0.90 interval, full-length development CE rises by **0.0134 ± 0.0014** nats/token and controller KL rises by **1.0657 ± 0.0329**. The KL increment is about **1.20×** the previous 0.70-to-0.80 increment. This is mild acceleration, not evidence of an abrupt phase transition.

The branch mismatch is effectively static across alpha because the frozen attention output and the trained Mamba output do not themselves change when the gate changes. At alpha 0.80, the Mamba branch has an RMS ratio near 0.92 relative to source attention, an absolute featurewise log-variance-ratio diagnostic near 1.08, and token-Gram relative error near 0.12. Increasing alpha linearly exposes that existing mismatch to the frozen downstream network.

Downstream residual drift is larger after layer 2 than immediately after the replaced layer. At alpha 0.90 versus the 0.80 reference, post-layer-1 relative drift is **0.0166 ± 0.0003**, whereas post-layer-2 drift is **0.0290 ± 0.0015**. This supports a bounded statement of downstream amplification: a small interface change is amplified by the following frozen block before reaching the output head.

## Important Measurement Caveat

The controller KL used `batchmean` reduction. That quantity scales with the number of predicted tokens and is therefore not directly comparable between the 24- and 48-token views. Experiment 016 does **not** establish a context-length-dependent failure. A future controller must report both its legacy batchmean diagnostic and a token-normalized KL quantity.

## Single Constrained Protocol Hypothesis for a Future Causal Test

The next causal candidate, if any, is a **featurewise output-moment calibrator** placed after the Mamba branch:

> `M_cal(h) = gamma ⊙ M(h) + beta`

where `gamma` starts at one and `beta` starts at zero. During calibration only, the functional condition receives an explicit source-attention moment objective for per-feature mean and variance. Every condition—including CE-only—receives the identical diagonal calibrator, parameter count, initialization, update budget, data, and gate schedule.

This is the narrowest intervention consistent with the observed static RMS and feature-variance mismatch. It is not yet validated. A fresh pre-registered study must also use token-normalized source-logit KL for gate monitoring, retain an untouched test split, and include CE-only plus value-only functional controls.

## Prohibitions

This development-only decision does not authorize a third layer, 7B scaling, a pure-Mamba claim, or a claim that pretraining can be avoided. It only identifies the next falsifiable interface-calibration hypothesis.
