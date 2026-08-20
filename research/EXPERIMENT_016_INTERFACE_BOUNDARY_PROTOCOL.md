# Experiment 016: Development-Only Diagnosis of the Layer-1 Gate Boundary

## Purpose

Experiments 014 and 015 established a repeated contradiction. Teacher-guided objectives reduced local value and directional mismatch, yet CE-only adaptation gave the best completed two-layer language-model endpoint. In Experiment 015, the adaptive controller accepted layer-1 `α=0.80` but rejected the `α=0.90` and `α=1.00` targets in every seed because source-logit KL exceeded the locked safety bound.

Experiment 016 is **not another conversion trial**. It is a protocol-development measurement study intended to identify what changes at the interface between `α=0.80` and `α=0.90`. It will neither calculate an untouched-test metric nor make a scaling claim.

## Frozen Inputs and Reconstruction

The study reconstructs the Experiment 015 adaptive-interface condition from the same source checkpoint, calibration texts, development partition, seeds, initialization, model configuration, adapter rank, objectives, optimizer settings, and training budget. It records the reproduced controller endpoint but does not modify the objective, controller, or model parameters after reconstruction.

The frozen source is `HuggingFaceTB/SmolLM-135M`. Layer 0 is fully gated to `α=1`; the diagnostic sweep changes only layer 1. The study uses the Experiment 015 seeds `20260841` through `20260845`.

## Data Guardrails

| Partition | Use in Experiment 016 |
|---|---|
| WikiText-2 raw train | Reconstruct the original adaptive-condition optimization only |
| WikiText-2 raw validation | All boundary measurements and summaries |
| WikiText-2 raw test | **Prohibited**: not loaded, decoded, scored, inspected, or used for model selection |

The primary unit of analysis is the fixed 64-sequence development subset. The protocol must fail if any test split is loaded by the diagnostic script.

## Gate Sweep

After reconstruction, the layer-1 Mamba and adapters are frozen. The model is measured at gate values:

> `α ∈ {0.00, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00}`

The `α=0.80` state is the reference for boundary deltas. Values on both sides of the boundary distinguish a sharp interface failure from smooth degradation.

## Measurements

All metrics are development-only and are aggregated over sequences before cross-seed summaries.

| Family | Measurement | Diagnostic interpretation |
|---|---|---|
| Global language behavior | Token CE loss; teacher–hybrid logit KL; logit relative-L2 change from `α=0.80` | Determines whether the output failure begins sharply at the gate boundary |
| Branch value alignment | Mamba-versus-attention relative-L2; cosine; global mean shift; RMS ratio | Tests whether source and replacement values diverge in scale or orientation |
| Branch geometry | Centered per-feature variance-ratio error; token-Gram relative error | Tests whether second-order feature/token geometry diverges despite low local NMSE |
| Residual stream | Mean, RMS, standard deviation, and relative-L2/cosine change of the post-layer-1 and post-layer-2 hidden states from `α=0.80` | Locates whether mismatch is amplified after the replacement interface |
| Sequence sensitivity | Repeat CE and KL summaries at 24- and 48-token views of the same development examples | Tests whether the boundary depends on state/context length rather than only layerwise distribution mismatch |
| Gate controller trace | Accepted gate, loss increment, and KL at each original target | Confirms that the reconstructed boundary matches Experiment 015 |

## Hypotheses and Falsifiers

| Hypothesis | Predicted signature | Falsifier |
|---|---|---|
| H1: Scale or mean mismatch | Branch mean/RMS or variance-ratio error steepens around 0.85–0.90 and tracks KL | KL rises without material moment shift |
| H2: Higher-order geometry mismatch | Token-Gram error rises near the boundary while first/second scalar moments remain stable | Token-Gram error is flat across the sweep |
| H3: Downstream amplification | Post-layer-2 drift grows faster than post-layer-1 drift and tracks logit KL | Downstream hidden states remain stable while logits diverge |
| H4: Context/state sensitivity | 48-token KL rise exceeds the 24-token rise near the boundary | Boundary is equally strong at both lengths |
| H5: Smooth interpolation, not a true cliff | All metrics change approximately linearly with alpha | Accelerating slope or discontinuity near 0.85–0.90 |

## Interpretation Rules

Experiment 016 is descriptive. It may nominate one **development-only protocol hypothesis** for a future locked causal test, but it cannot validate an intervention or claim test-set improvement. No third layer, standalone Mamba, 7B scale-up, or claim about avoiding pretraining is permitted based on these data.

A future Experiment 017 is allowed only after the results select one specific, equal-parameter intervention plus CE-only control, with fresh held-out final evaluation and independent seeds locked before execution.
