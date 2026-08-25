# Experiment 021A Development-Only Aggregate

**Scope:** Five-seed sensitivity-basis feasibility pilot. The test split was prohibited and `test_split_requested` is false in every result. These are development diagnostics, not endpoint transfer evidence.

## Per-Seed Differences

Positive CE difference means worse than the unit-basis CE control. Negative discrepancy difference means lower sensitivity-weighted attention-output discrepancy.

| Seed | Correct CE − unit | Correct weighted − unit | Correct weighted − permuted |
|---:|---:|---:|---:|
| 20260901 | -0.13894 | -0.01782 | -0.03764 |
| 20260902 | -0.03627 | 0.00917 | -0.01623 |
| 20260903 | -0.06654 | -0.06450 | -0.11644 |
| 20260904 | -0.04966 | -0.11260 | -0.17684 |
| 20260905 | -0.04768 | -0.05238 | -0.21644 |

## Locked Feasibility Rule

| Requirement | Result | Assessment |
|---|---:|---|
| No test access | 5/5 test flags false | **Pass** |
| Alpha-zero integrity | maximum deviation 0.00e+00 | **Pass** |
| Finite alpha-one endpoints | 15/15 endpoints | **Pass** |
| Correct basis CE safety | mean B−A = -0.06782 | **Pass** |
| Correct basis lower weighted discrepancy than unit | mean B−A = -0.04763 | **Pass** |
| Correct basis lower weighted discrepancy than permuted | mean B−C = -0.11272 | **Pass** |

**A separately preregistered endpoint protocol is permitted.** This does not establish transfer success and cannot alter the fixed basis formula.

## Development Diagnostics

| Condition | Development CE | Weighted discrepancy | Unweighted MSE | Post-layer-2 drift |
|---|---:|---:|---:|---:|
| Unit CE transport | 3.72366 ± 0.02298 | 0.97160 ± 0.07425 | 1.04348 ± 0.06994 | 0.50461 ± 0.01341 |
| Correct sensitivity basis | 3.65584 ± 0.04950 | 0.92397 ± 0.03571 | 0.97178 ± 0.03916 | 0.50620 ± 0.00401 |
| Permuted sensitivity basis | 3.69786 ± 0.01780 | 1.03669 ± 0.10546 | 1.10017 ± 0.11022 | 0.51192 ± 0.01974 |
