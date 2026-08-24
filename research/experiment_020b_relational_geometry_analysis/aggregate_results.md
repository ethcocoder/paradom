# Experiment 020B Aggregate Results

**Scope:** Five paired endpoint seeds. The fresh WikiText-2 test slice contains eligible sequences 513–640 and was requested only after all three conditions completed their 720 updates in every seed.

## Fresh Primary Endpoint

Positive correct-relational advantages favor the correctly indexed relational-geometry condition.

| Seed | Teacher loss | CE loss | Correct relation loss | Permuted relation loss | Correct vs CE | Correct vs permuted |
|---:|---:|---:|---:|---:|---:|---:|
| 20260891 | 4.1613 | 4.4035 | 4.4133 | 4.4383 | -0.0097 | 0.0250 |
| 20260892 | 4.1613 | 4.4441 | 4.4219 | 4.4232 | 0.0221 | 0.0012 |
| 20260893 | 4.1613 | 4.4506 | 4.4265 | 4.3765 | 0.0241 | -0.0499 |
| 20260894 | 4.1613 | 4.3789 | 4.3810 | 4.3808 | -0.0021 | -0.0002 |
| 20260895 | 4.1613 | 4.3785 | 4.3767 | 4.3848 | 0.0018 | 0.0081 |
| **Mean ± sample SD** | **4.1613 ± 0.0000** | **4.4111 ± 0.0347** | **4.4039 ± 0.0234** | **4.4007 ± 0.0281** | **0.0072 ± 0.0151** | **-0.0032 ± 0.0280** |

## Diagnostics and Acceptance Checks

| Criterion | Result | Assessment |
|---|---:|---|
| Alpha-zero exactness, alpha-one endpoints, and final-slice isolation | 5/5; maximum deviation 0.00e+00 | **Pass** |
| Correct relation wins versus CE | 3/5; mean advantage 0.0072 | **Fail** |
| Correct relation wins versus permuted relation | 3/5; mean advantage -0.0032 | **Fail** |
| Correct relation ECE increase versus CE <= 0.01 | maximum 0.0058 | **Pass** |
| Correct relation within 0.15 loss of teacher | maximum gap 0.2652 | **Fail** |
| Third layer / scaling / quantization authorization | — | **Denied** |

## Development-Only Mechanism Diagnostics

| Condition | Development correct-relation KL | Development CE | Post-layer-1 drift | Post-layer-2 drift |
|---|---:|---:|---:|---:|
| CE transport | 0.0271 ± 0.0016 | 3.6797 ± 0.0370 | 0.4922 ± 0.0335 | 0.4846 ± 0.0200 |
| Correct relation | 0.0221 ± 0.0015 | 3.6922 ± 0.0393 | 0.4871 ± 0.0358 | 0.4853 ± 0.0216 |
| Permuted relation | 0.0211 ± 0.0012 | 3.6765 ± 0.0814 | 0.4690 ± 0.0288 | 0.4804 ± 0.0165 |

## Fixed Scientific-Prompt Continuations

### Seed 20260891

| Condition | Continuation |
|---|---|
| CE transport | The purpose of scientific research is to advance our knowledge of the natural world.<br>The scientific method is a systematic approach |
| Correct relation | The purpose of scientific research is to advance our knowledge of the natural world.<br>The scientific method is a systematic method |
| Permuted relation | The purpose of scientific research is to advance our understanding of the world.<br>The scientific community is a diverse group of |

### Seed 20260892

| Condition | Continuation |
|---|---|
| CE transport | The purpose of scientific research is to advance our knowledge of the natural world.<br>The scientific method of research is to |
| Correct relation | The purpose of scientific research is to advance our knowledge of the natural world.<br>The scientific method of research is to |
| Permuted relation | The purpose of scientific research is to advance our knowledge of the natural world.<br>The scientific method of research is to |

### Seed 20260893

| Condition | Continuation |
|---|---|
| CE transport | The purpose of scientific research is to advance our knowledge of the natural world. The scientific method of research is based on |
| Correct relation | The purpose of scientific research is to advance our knowledge of the natural world. The scientific method of research is to obtain |
| Permuted relation | The purpose of scientific research is to advance our knowledge of the natural world. The scientific method of research is a method |

### Seed 20260894

| Condition | Continuation |
|---|---|
| CE transport | The purpose of scientific research is to find the truth of the universe.<br>The scientific method of research is the scientific |
| Correct relation | The purpose of scientific research is to find the truth of the universe.<br>The scientific method of research is to find |
| Permuted relation | The purpose of scientific research is to advance our understanding of the natural world.<br>The scientific method of research is to |

### Seed 20260895

| Condition | Continuation |
|---|---|
| CE transport | The purpose of scientific research is to advance our knowledge of the universe.<br>The scientific method of research is based on |
| Correct relation | The purpose of scientific research is to advance our knowledge of the universe.<br>The scientific community is divided into two major |
| Permuted relation | The purpose of scientific research is to advance our understanding of the natural world.<br>The scientific method of research is a |
