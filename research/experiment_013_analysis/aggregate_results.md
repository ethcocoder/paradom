# Experiment 013: Aggregated Replication Results

## Held-Out Test Loss by Seed

| Seed | Teacher test loss | Direct Mamba test loss | Random Mamba test loss | Direct − random | Random − teacher | α=0 absolute error |
|---:|---:|---:|---:|---:|---:|---:|
| 20260821 | 3.8146 | 4.0579 | 3.9128 | 0.1451 | 0.0982 | 0.00e+00 |
| 20260822 | 3.8146 | 4.0674 | 3.9166 | 0.1508 | 0.1020 | 0.00e+00 |
| 20260823 | 3.8146 | 4.0460 | 3.9392 | 0.1068 | 0.1246 | 0.00e+00 |

## Mean ± Sample Standard Deviation

| Metric | Result |
|---|---:|
| Teacher final test loss | 3.8146 ± 0.0000 |
| Direct initialization final test loss | 4.0571 ± 0.0107 |
| Random initialization final test loss | 3.9229 ± 0.0143 |
| Direct − random loss (paired) | 0.1342 ± 0.0239 |
| Random − teacher loss | 0.1083 ± 0.0143 |
| Direct layer-1 NMSE (development) | 0.2258 ± 0.0014 |
| Random layer-1 NMSE (development) | 0.1481 ± 0.0028 |

## Mechanical Acceptance Checks

| Check | Result |
|---|---|
| α=0 preserves teacher exactly on final test in every seed | True |
| Both final hybrid losses are finite in every seed | True |
| Random hybrid final test loss < 3.5 in every seed | False |
| Random initialization beats direct mapping in every seed | True |

## Fixed-Prompt Continuations for Human Review

| Seed | Condition | Scientific research prompt | Explorer prompt |
|---:|---|---|---|
| 20260821 | Direct | The purpose of scientific research is to find out the causes of disease and to | Once upon a time, a young explorer discovered a new world.
The first time |
| 20260821 | Random | The purpose of scientific research is to find out the truth about the world. | Once upon a time, a young explorer discovered a new land, he was not only |
| 20260822 | Direct | The purpose of scientific research is to find out the causes of disease and to | Once upon a time, a young explorer discovered a new world.
The first time |
| 20260822 | Random | The purpose of scientific research is to find out how the world works.
 | Once upon a time, a young explorer discovered a new world.
The first time |
| 20260823 | Direct | The purpose of scientific research is to find out the causes of disease and to | Once upon a time, a young explorer discovered a new world.
The first time |
| 20260823 | Random | The purpose of scientific research is to find out the truth about the world. | Once upon a time, a young explorer discovered a new world, and he was not |

The loss comparison is a three-seed descriptive replication rather than a powered significance test. English quality must be judged from the displayed continuations rather than inferred from loss alone.
