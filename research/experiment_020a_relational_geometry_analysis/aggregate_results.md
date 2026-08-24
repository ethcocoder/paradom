# Experiment 020A Development-Only Aggregate

**Scope:** Five-seed coefficient-selection pilot. The script loaded only WikiText-2 train and validation data; every result explicitly records `test_split_requested: false`. These are development diagnostics, not held-out transfer results.

## Per-Seed Development Differences

Positive CE delta is worse than the conditional-CE transport control. Negative relation delta indicates lower post-block relation KL than that control.

| Seed | λ=0.05 CE Δ | λ=0.10 CE Δ | λ=0.20 CE Δ | λ=0.05 relation Δ | λ=0.10 relation Δ | λ=0.20 relation Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 20260881 | 0.00102 | -0.00059 | -0.00699 | -0.00215 | -0.00416 | -0.00731 |
| 20260882 | 0.01260 | 0.01462 | 0.01633 | -0.00215 | -0.00375 | -0.00608 |
| 20260883 | -0.00650 | -0.01121 | -0.03089 | -0.00115 | -0.00273 | -0.00530 |
| 20260884 | 0.00108 | 0.00445 | 0.01186 | -0.00159 | -0.00297 | -0.00530 |
| 20260885 | -0.00592 | -0.00372 | 0.00010 | -0.00212 | -0.00402 | -0.00605 |

## Locked Coefficient Selection

| Coefficient | Mean development CE Δ ± sample SD | Mean relation-KL Δ ± sample SD | Maximum alpha-zero loss deviation | All finite / alpha-one | Eligible |
|---:|---:|---:|---:|---|---|
| 0.05 | 0.00046 ± 0.00770 | -0.00183 ± 0.00045 | 0.00e+00 | yes | **yes** |
| 0.10 | 0.00071 ± 0.00963 | -0.00353 ± 0.00064 | 0.00e+00 | yes | **yes** |
| 0.20 | -0.00192 ± 0.01865 | -0.00601 ± 0.00082 | 0.00e+00 | yes | **yes** |

**Selected coefficient: λ=0.20.** It is the largest eligible coefficient under the locked development-only rule. This selection does not establish an endpoint benefit and does not access the test partition.
