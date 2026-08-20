# Experiment 014: Five-Seed Causal-Control Aggregate

## Frozen Held-Out Test Results

| Seed | Teacher loss | Functional loss | CE-only loss | CE-only − functional | Functional − teacher |
|---:|---:|---:|---:|---:|---:|
| 20260831 | 4.0221 | 4.1389 | 4.1296 | -0.0093 | 0.1168 |
| 20260832 | 4.0221 | 4.1357 | 4.0424 | -0.0933 | 0.1136 |
| 20260833 | 4.0221 | 4.1299 | 4.0670 | -0.0629 | 0.1078 |
| 20260834 | 4.0221 | 4.1314 | 4.0999 | -0.0315 | 0.1093 |
| 20260835 | 4.0221 | 4.1368 | 4.1228 | -0.0139 | 0.1147 |

## Mean ± Sample Standard Deviation

| Metric | Value |
|---|---:|
| Teacher test loss | 4.0221 ± 0.0000 |
| Functional-distillation test loss | 4.1345 ± 0.0038 |
| CE-only test loss | 4.0923 ± 0.0371 |
| CE-only − functional loss | -0.0422 ± 0.0355 |
| Bootstrap 95% interval for paired difference | [-0.0713, -0.0156] |
| Functional layer-1 NMSE (development) | 0.1489 ± 0.0027 |
| CE-only layer-1 NMSE (development, post-hoc only) | 1.1650 ± 0.0528 |

A positive paired value favors functional distillation. The bootstrap interval is descriptive because the experiment has only five seeds.

## Fixed-Prompt Continuations for Human Review

| Seed | Condition | Scientific-research prompt | Explorer prompt |
|---:|---|---|---|
| 20260831 | Functional | The purpose of scientific research is to find out the truth about the world. The scientific method is a method of | Once upon a time, a young explorer discovered a large, flat, flat-topped, flat-bottomed, flat |
| 20260831 | CE-only | The purpose of scientific research is to find new and better ways to improve the quality of life. The scientific method | Once upon a time, a young explorer discovered a new and unique species of fish, the fish was named the “Giant |
| 20260832 | Functional | The purpose of scientific research is to find out the truth about the world. The scientific method is a systematic approach | Once upon a time, a young explorer discovered a new world, and he was able to make a new discovery. The |
| 20260832 | CE-only | The purpose of scientific research is to find new and useful knowledge. The scientific method is a systematic approach to the | Once upon a time, a young explorer discovered the island, and the islanders were called the “Sirens.” The |
| 20260833 | Functional | The purpose of scientific research is to find out how the world works. The purpose of scientific research is to find | Once upon a time, a young explorer discovered a new world, and he was not only a child. He was a man |
| 20260833 | CE-only | The purpose of scientific research is to improve the quality of life for people. The term “scientific research” is | Once upon a time, a young explorer discovered a new and beautiful island. The island was named after him. The island was |
| 20260834 | Functional | The purpose of scientific research is to find out the causes of disease and to develop new drugs and treatments. The | Once upon a time, a young explorer discovered a new world, he was able to make a new discovery. The first |
| 20260834 | CE-only | The purpose of scientific research is to improve the quality of life. The scientific research is a process of discovery, | Once upon a time, a young explorer discovered the island of the same name, and the island was named after him.  |
| 20260835 | Functional | The purpose of scientific research is to find out the truth about the world. The scientific method is a systematic approach | Once upon a time, a young explorer discovered a new land, he was able to make a new discovery. The first |
| 20260835 | CE-only | The purpose of scientific research is to advance our knowledge of the world. The scientific method is a systematic approach to | Once upon a time, a young explorer discovered the island of the same name, and named it after the king of the island |
