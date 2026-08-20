# Experiment 016: Development-Only Gate-Boundary Aggregate

## Guardrail

Only WikiText-2 train and validation partitions were requested. The frozen test split was not loaded or scored. All numbers below are development diagnostics, not generalization results.

## Mean Development Curves

| Alpha | CE (48) | KL (48) | CE (24) | KL (24) | Post-L1 drift from 0.80 | Post-L2 drift from 0.80 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 3.5739 ± 0.0031 | 4.0712 ± 0.0420 | 4.1023 ± 0.0049 | 2.6497 ± 0.0452 | 0.1318 | 0.1922 |
| 0.70 | 3.5949 ± 0.0057 | 6.3834 ± 0.0765 | 4.1218 ± 0.0095 | 3.7099 ± 0.0476 | 0.0164 | 0.0272 |
| 0.75 | 3.6001 ± 0.0062 | 6.8075 ± 0.0827 | 4.1274 ± 0.0102 | 3.9143 ± 0.0543 | 0.0082 | 0.0138 |
| 0.80 | 3.6059 ± 0.0068 | 7.2719 ± 0.0894 | 4.1338 ± 0.0110 | 4.1391 ± 0.0616 | 0.0000 | 0.0000 |
| 0.85 | 3.6123 ± 0.0074 | 7.7803 ± 0.0973 | 4.1409 ± 0.0117 | 4.3856 ± 0.0699 | 0.0083 | 0.0142 |
| 0.90 | 3.6193 ± 0.0080 | 8.3377 ± 0.1069 | 4.1488 ± 0.0123 | 4.6574 ± 0.0798 | 0.0166 | 0.0290 |
| 0.95 | 3.6271 ± 0.0086 | 8.9500 ± 0.1176 | 4.1577 ± 0.0129 | 4.9593 ± 0.0901 | 0.0250 | 0.0441 |
| 1.00 | 3.6357 ± 0.0092 | 9.6224 ± 0.1307 | 4.1676 ± 0.0135 | 5.2963 ± 0.1010 | 0.0336 | 0.0597 |

## Boundary Delta: 0.90 − 0.80

| Metric | Mean ± sample SD |
|---|---:|
| full 48 ce 090 minus 080 | 0.0134 ± 0.0014 |
| full 48 logit kl 090 minus 080 | 1.0657 ± 0.0329 |
| short 24 ce 090 minus 080 | 0.0150 ± 0.0017 |
| short 24 logit kl 090 minus 080 | 0.5183 ± 0.0267 |
| full 48 post layer1 relative l2 from 080 090 minus 080 | 0.0166 ± 0.0003 |
| full 48 post layer2 relative l2 from 080 090 minus 080 | 0.0290 ± 0.0015 |
| full 48 branch rms ratio 090 minus 080 | 0.0000 ± 0.0000 |
| full 48 branch feature log variance ratio abs 090 minus 080 | 0.0000 ± 0.0000 |
| full 48 branch token gram relative l2 090 minus 080 | 0.0000 ± 0.0000 |

## Controller Reproduction

| Seed | Accepted L1 alpha | KL at 0.80 | KL at 0.90 | 0.90 accepted? |
|---:|---:|---:|---:|---|
| 20260841 | 0.80 | 7.4501 | 8.3154 | False |
| 20260842 | 0.80 | 7.4527 | 8.3902 | False |
| 20260843 | 0.80 | 7.4748 | 8.3475 | False |
| 20260844 | 0.80 | 7.5978 | 8.5643 | False |
| 20260845 | 0.80 | 7.5907 | 8.5341 | False |
