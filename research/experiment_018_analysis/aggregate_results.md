# Experiment 018 Aggregate Results

**Scope:** Five paired seeds, three equal-parameter residual-interface transport conditions, and a fresh held-out WikiText-2 test slice (eligible test sequences 257–384). The final split was loaded only after all conditions completed their optimization in each seed.

## Fresh-Test Loss by Seed

| Seed | Teacher | Unconditional CE | Conditional CE | Conditional anchor | Conditional advantage (U − C) | Anchor advantage (C − A) |
|---:|---:|---:|---:|---:|---:|---:|
| 20260861 | 4.0162 | 4.2140 | 4.1712 | 4.1690 | 0.0428 | 0.0022 |
| 20260862 | 4.0162 | 4.2142 | 4.1501 | 4.1465 | 0.0641 | 0.0036 |
| 20260863 | 4.0162 | 4.2524 | 4.2508 | 4.2451 | 0.0016 | 0.0056 |
| 20260864 | 4.0162 | 4.2650 | 4.2300 | 4.2228 | 0.0350 | 0.0072 |
| 20260865 | 4.0162 | 4.1902 | 4.1944 | 4.1960 | -0.0042 | -0.0016 |
| **Mean ± sample SD** | **4.0162 ± 0.0000** | **4.2271 ± 0.0307** | **4.1993 ± 0.0413** | **4.1959 ± 0.0397** | **0.0279 ± 0.0287** | **0.0034 ± 0.0034** |

A positive conditional advantage favors the conditional transport map. A positive anchor advantage favors the downstream-anchor condition.

## Development Diagnostics after Layer 1

| Condition | Development CE | Token-normalized source-logit KL | Correction RMS / Mamba RMS | Downstream-anchor diagnostic |
|---|---:|---:|---:|---:|
| Unconditional CE transport | 3.7474 ± 0.0274 | 0.8365 ± 0.0275 | 0.2631 ± 0.0745 | — |
| Conditional CE transport | 3.7190 ± 0.0336 | 0.7941 ± 0.0312 | 0.2178 ± 0.0770 | — |
| Conditional anchor transport | 3.7148 ± 0.0335 | 0.7884 ± 0.0318 | 0.2202 ± 0.0759 | 0.0613 ± 0.0016 |

## Predeclared Criteria

| Criterion | Result | Assessment |
|---|---:|---|
| Exact alpha-zero source preservation | 5/5 seeds | **Pass** |
| Complete two-layer alpha-one endpoints | 5/5 seeds | **Pass** |
| Fresh-test isolation after all conditions train | 5/5 seeds | **Pass** |
| Conditional CE wins versus unconditional CE | 4/5; mean advantage 0.0279 | **Fail** |
| Downstream anchor wins versus conditional CE | 4/5; mean advantage 0.0034 | **Fail** |
| Anchor mean gap to frozen teacher <= 0.15 | 0.1796 | **Fail** |
| Permission for third layer | — | **Denied** |

## Fixed-Prompt Continuations

### Seed 20260861

| Condition | Scientific-prompt continuation |
|---|---|
| Unconditional CE transport | The purpose of scientific research is to advance our understanding of the natural world. The scientific method of research is based on |
| Conditional CE transport | The purpose of scientific research is to advance our knowledge of the natural world.
The scientific method is a systematic method |
| Conditional anchor transport | The purpose of scientific research is to advance our knowledge of the natural world.
The scientific method is a systematic method |

### Seed 20260862

| Condition | Scientific-prompt continuation |
|---|---|
| Unconditional CE transport | The purpose of scientific research is to discover and to explain the laws of nature.
The scientific method is a systematic |
| Conditional CE transport | The purpose of scientific research is to advance our knowledge of the natural world.
The scientific method is the scientific method |
| Conditional anchor transport | The purpose of scientific research is to advance our knowledge of the natural world.
The scientific method is a systematic method |

### Seed 20260863

| Condition | Scientific-prompt continuation |
|---|---|
| Unconditional CE transport | The purpose of scientific research is to advance our knowledge of the natural world. The scientific method is a systematic method of |
| Conditional CE transport | The purpose of scientific research is to advance our understanding of the natural world. The purpose of scientific research is to advance |
| Conditional anchor transport | The purpose of scientific research is to advance our understanding of the natural world. The scientific method is a systematic approach to |

### Seed 20260864

| Condition | Scientific-prompt continuation |
|---|---|
| Unconditional CE transport | The purpose of scientific research is to provide a better understanding of the world.
The scientific method of research is a |
| Conditional CE transport | The purpose of scientific research is to find and understand the nature of the universe.
The scientific method of observation is |
| Conditional anchor transport | The purpose of scientific research is to find and understand the nature of the universe.
The scientific method of observation is |

### Seed 20260865

| Condition | Scientific-prompt continuation |
|---|---|
| Unconditional CE transport | The purpose of scientific research is to advance our knowledge of the natural world.
The scientific method is a systematic approach |
| Conditional CE transport | The purpose of scientific research is to find a solution to a problem.
The problem is usually a scientific or technical |
| Conditional anchor transport | The purpose of scientific research is to find a solution to a problem.
The problem is usually a scientific or technical |

