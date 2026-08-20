# Experiment 014: Causal Test of Teacher-Guided Functional Distillation

## Objective

Experiment 013 established that a randomly initialized Mamba branch can replace two frozen SmolLM attention outputs with stable English behavior, and that direct static tensor mapping is worse than fresh Mamba initialization followed by functional fitting. Experiment 014 asks the missing causal question: **does teacher-guided functional distillation provide a measurable advantage over ordinary next-token fitting on the same small calibration corpus and optimization budget?**

## Fixed Model and Architecture

The frozen source and retained hybrid backbone are `HuggingFaceTB/SmolLM-135M`. Attention layers 0 and 1 are replaced sequentially by fresh Mamba mixers with state sizes 64 and 96. Embeddings, MLPs, norms, output head, all non-replaced attention layers, and the source-attention modules retained inside wrappers are frozen. At the endpoint, both replacement gates are fixed to `α=1`, so their Mamba branches are solely responsible for the former attention outputs.

## Matched Conditions

| Condition | Initialization | Local phase, 540 updates | Gate phase, 180 updates | Teacher outputs or logits available to optimizer? |
|---|---|---|---|---|
| Functional distillation | Fresh random Mamba | `L_local` against frozen source attention trajectories | `0.70 KL + 0.20 CE + 0.10 L_local` with the shared gate schedule | Yes |
| CE-only control | The identical fresh random Mamba initialization within seed | Next-token CE at α=1; no trajectory target | Next-token CE with the shared gate schedule | No |

Both conditions use the same Mamba architecture, layer order, learning rates, optimizer, weight decay, maximum sequence length, 1,024 calibration sequences, and exactly 720 optimizer updates per condition. The random-number generator is reset to the same seed before building each condition, so both start from identical Mamba parameters. In the CE-only local phase, α is set to 1 so next-token loss has a nonzero gradient with respect to the Mamba parameters. The gate phase uses α=0.05 for its first ten warm-up updates and then rises linearly to 1 for **both** conditions; this small common nonzero floor is necessary because CE alone cannot train a fully gated-off mixer. Exact α=0 equivalence to the teacher is checked separately before any fitting. The CE-only condition intentionally gives the student the natural self-supervised language-model signal but removes all teacher-derived activations and logits.

## Data Separation

| Purpose | WikiText-2 raw partition | Sample size | Allowed use |
|---|---|---:|---|
| Calibration | Train | 1,024 usable sequences | Optimizer inputs only |
| Development diagnostics | Validation | 64 usable sequences | Monitoring only; no optimizer inputs |
| Final held-out evaluation | Test | 128 usable sequences | Final loss, perplexity, and fixed-prompt generation only |

The first usable sequences in each named split are deterministically selected after tokenizer filtering. The test sequence list, maximum length, and all seeds are frozen before executing the first run. The validation partition is never optimized against, and the test partition is never inspected during fitting or gate scheduling.

## Independent Runs

Five complete paired runs will use seeds `20260831`, `20260832`, `20260833`, `20260834`, and `20260835`. Every seed runs both conditions. Only Mamba initialization and optimizer randomness vary across seeds. Results are compared within seed as `CE-only loss − functional-distillation loss`, where a positive value favors functional distillation.

## Primary Outcome and Acceptance Rules

The primary outcome is final token-weighted causal-language-model loss on the frozen 128-sequence test list after both layer gates equal 1. The experiment is successful as causal evidence only if all of the following hold:

| Criterion | Rule |
|---|---|
| Endpoint integrity | α=0 branch loss equals the frozen teacher loss to within `1e-5` for all seeds |
| Numerical stability | Both conditions produce finite final test loss in all seeds |
| Causal advantage | Functional distillation has lower final test loss than CE-only in at least 4 of 5 paired seeds and a positive mean paired difference of at least 0.05 nats/token |
| Behavioral stability | Functional condition has coherent English continuations on the scientific-research and explorer prompts in at least 4 of 5 seeds |
| Scaling readiness | Functional condition’s mean final test loss is no worse than 0.15 nats/token above the frozen teacher, and it beats the CE-only condition by the causal-advantage rule |

The direct static-map route is deliberately not repeated because Experiment 013 showed it loses to random initialization in 3/3 paired runs. Its result remains a documented negative control, not a candidate transfer mechanism.

## Interpretation Guardrails

A functional-distillation advantage would demonstrate that teacher trajectories and teacher logits contribute useful information beyond ordinary calibration-text CE fitting under the matched architecture, data, and step budget. It would still not demonstrate pure Mamba conversion, erase the contribution of frozen retained SmolLM components, or establish an end-to-end pretraining cost comparison. Any third-layer experiment must be conditional on passing the scaling-readiness rule above.
