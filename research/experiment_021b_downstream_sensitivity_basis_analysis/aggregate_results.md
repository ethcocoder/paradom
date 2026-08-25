# Experiment 021B Aggregate Results

**Scope:** Five paired endpoint seeds. The new WikiText-2 fresh final range is eligible test sequences 641–768, loaded only after all three conditions trained in every seed.

## Fresh Primary Endpoint

Positive advantages favor correctly ordered downstream-sensitivity basis B.

| Seed | Teacher loss | A: Unit CE | B: Correct basis | C: Permuted basis | A − B | C − B |
|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 3.9184 | 4.0820 | 4.0638 | 4.0755 | 0.0182 | 0.0116 |
| 20260912 | 3.9184 | 4.0546 | 4.0749 | 4.0429 | -0.0203 | -0.0320 |
| 20260913 | 3.9184 | 4.0739 | 4.0560 | 4.1151 | 0.0178 | 0.0591 |
| 20260914 | 3.9184 | 4.0451 | 4.0790 | 4.0385 | -0.0340 | -0.0405 |
| 20260915 | 3.9184 | 4.0367 | 4.0306 | 4.0344 | 0.0061 | 0.0038 |
| **Mean ± sample SD** | **3.9184 ± 0.0000** | **4.0584 ± 0.0191** | **4.0609 ± 0.0192** | **4.0613 ± 0.0342** | **-0.0024 ± 0.0236** | **0.0004 ± 0.0397** |

## Acceptance Checks

| Criterion | Result | Assessment |
|---|---:|---|
| Alpha-zero integrity, alpha-one completion, and final-slice isolation | 5/5; maximum deviation 0.00e+00 | **Pass** |
| Correct basis wins versus unit | 3/5; mean advantage -0.0024 | **Fail** |
| Correct basis wins versus permuted | 3/5; mean advantage 0.0004 | **Fail** |
| Correct-basis ECE increase versus unit <= 0.01 | maximum 0.0036 | **Pass** |
| Correct basis within 0.15 loss of teacher | maximum gap 0.1606 | **Fail** |
| Third layer / scaling / quantization | All prior criteria required | **Denied** |

## Development Mechanism Diagnostics

| Condition | Development CE | Weighted discrepancy | Unweighted MSE | Post-layer-2 drift |
|---|---:|---:|---:|---:|
| Unit CE transport | 3.6630 ± 0.0307 | 0.9989 ± 0.0559 | 1.0732 ± 0.0630 | 0.4961 ± 0.0055 |
| Correct sensitivity basis | 3.6734 ± 0.0491 | 0.9812 ± 0.0883 | 1.0285 ± 0.0753 | 0.5142 ± 0.0242 |
| Permuted sensitivity basis | 3.6619 ± 0.0475 | 1.0288 ± 0.0669 | 1.1010 ± 0.0714 | 0.4950 ± 0.0101 |

## Fixed Scientific-Prompt Continuations

### Seed 20260911

| Condition | Continuation |
|---|---|
| Unit CE transport | The purpose of scientific research is to find a solution to a problem.<br>The problem is usually a problem of the |
| Correct sensitivity basis | The purpose of scientific research is to advance our knowledge of the natural world and to improve our understanding of the processes of |
| Permuted sensitivity basis | The purpose of scientific research is to advance our knowledge of the natural world. The purpose of scientific research is to advance |

### Seed 20260912

| Condition | Continuation |
|---|---|
| Unit CE transport | The purpose of scientific research is to advance our knowledge of the natural world.<br>The scientific method of research is based |
| Correct sensitivity basis | The purpose of scientific research is to advance our knowledge of the natural world.<br>The scientific method of research is to |
| Permuted sensitivity basis | The purpose of scientific research is to advance our knowledge of the natural world. The purpose of scientific research is to advance |

### Seed 20260913

| Condition | Continuation |
|---|---|
| Unit CE transport | The purpose of scientific research is to advance our understanding of the natural world.<br>The scientific method of research is to |
| Correct sensitivity basis | The purpose of scientific research is to find new and better ways to improve the quality of life on earth.<br>The |
| Permuted sensitivity basis | The purpose of scientific research is to advance our understanding of the natural world.<br>The scientific method of research is to |

### Seed 20260914

| Condition | Continuation |
|---|---|
| Unit CE transport | The purpose of scientific research is to find and to understand the nature of the universe.<br>The scientific method of research |
| Correct sensitivity basis | The purpose of scientific research is to advance our understanding of the natural world.<br>The scientific method of research is to |
| Permuted sensitivity basis | The purpose of scientific research is to advance our knowledge of the natural world.<br>The scientific method is the scientific method |

### Seed 20260915

| Condition | Continuation |
|---|---|
| Unit CE transport | The purpose of scientific research is to find and to understand the nature of the universe.<br>The universe is a vast |
| Correct sensitivity basis | The purpose of scientific research is to provide a better understanding of the world around us.<br>The scientific method of research |
| Permuted sensitivity basis | The purpose of scientific research is to find a solution to a problem.<br>The problem is usually a mathematical problem, |
