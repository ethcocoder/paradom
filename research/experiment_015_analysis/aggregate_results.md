# Experiment 015: Aggregate Results

## Endpoint and Held-Out Test Results

| Seed | CE-adapter test loss | Value-functional test loss | CE − value | Adaptive L0 alpha | Adaptive L1 alpha | Adaptive endpoint |
|---:|---:|---:|---:|---:|---:|---|
| 20260841 | 4.0726 | 4.1279 | -0.0552 | 1.00 | 0.80 | failed |
| 20260842 | 4.0686 | 4.1421 | -0.0735 | 1.00 | 0.80 | failed |
| 20260843 | 4.0663 | 4.1340 | -0.0677 | 1.00 | 0.80 | failed |
| 20260844 | 4.0630 | 4.1426 | -0.0796 | 1.00 | 0.80 | failed |
| 20260845 | 4.0532 | 4.1304 | -0.0772 | 1.00 | 0.80 | failed |

## Aggregate

| Metric | Mean ± sample SD |
|---|---:|
| CE-adapter test loss | 4.0647 ± 0.0073 |
| Value-functional test loss | 4.1354 ± 0.0067 |
| CE − value test loss | -0.0707 ± 0.0097 |
| Value-functional L1 directional NMSE | 1.0699 ± 0.0068 |
| Adaptive-interface L1 directional NMSE | 0.9912 ± 0.0031 |

Adaptive runs did not receive final test metrics because the locked endpoint rule was not met. This preserves the test protocol and prevents an alpha=0.8 partial hybrid from being presented as a two-layer replacement.

## Fixed-Prompt Continuations at Valid alpha=1 Endpoints

| Seed | CE-adapter science | Value-functional science |
|---:|---|---|
| 20260841 | The purpose of scientific research is to advance our understanding of the world around us.
Scientific research is a broad field | The purpose of scientific research is to find out the truth about the world.
The scientific method is a systematic approach |
| 20260842 | The purpose of scientific research is to provide a better understanding of the world around us.

The scientific method is | The purpose of scientific research is to find out the truth about the world.
The scientific method is a systematic approach |
| 20260843 | The purpose of scientific research is to advance our understanding of the world around us.
The scientific community is a diverse | The purpose of scientific research is to find out the truth about the world.
The scientific method is a systematic approach |
| 20260844 | The purpose of scientific research is to improve the quality of life.
The term " scientific research" is used to | The purpose of scientific research is to find out the truth about the world.
The scientific method is a systematic approach |
| 20260845 | The purpose of scientific research is to advance our knowledge of the world and to improve the quality of life.
The | The purpose of scientific research is to find out the truth about the world.
The scientific method is a systematic approach |
