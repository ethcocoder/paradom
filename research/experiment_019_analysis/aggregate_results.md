# Experiment 019 Aggregate Results

**Scope:** Five paired output-sharpening seeds. Every condition completed an identical 720-update deployed conditional-transport pre-stage and a 180-update output stage. The fresh WikiText-2 test slice consisted of eligible sequences 385–512 and was loaded only after all conditions completed both stages.

## Fresh-Test Quality, Sharpness, and Calibration

| Seed | Teacher loss | CE loss | Self-sharp loss | Teacher-sharp loss | Teacher advantage vs CE | Teacher advantage vs self |
|---:|---:|---:|---:|---:|---:|---:|
| 20260871 | 3.8438 | 3.9327 | 3.9251 | 3.9126 | 0.0201 | 0.0125 |
| 20260872 | 3.8438 | 3.8949 | 3.8949 | 3.8947 | 0.0002 | 0.0002 |
| 20260873 | 3.8438 | 3.9450 | 3.9418 | 3.9269 | 0.0181 | 0.0149 |
| 20260874 | 3.8438 | 3.9002 | 3.8945 | 3.8895 | 0.0107 | 0.0050 |
| 20260875 | 3.8438 | 3.9075 | 3.9003 | 3.8900 | 0.0174 | 0.0103 |
| **Mean ± sample SD** | **3.8438 ± 0.0000** | **3.9161 ± 0.0217** | **3.9113 ± 0.0212** | **3.9028 ± 0.0165** | **0.0133 ± 0.0081** | **0.0086 ± 0.0059** |

Positive teacher advantage favors the teacher-temperature-sharpening condition.

| Condition | Fresh token entropy | Fresh 20-bin top-token ECE | Mean loss gap to frozen teacher |
|---|---:|---:|---:|
| CE continuation | 3.8007 ± 0.0260 | 0.0240 ± 0.0033 | 0.0722 ± 0.0217 |
| Self-entropy sharpening | 3.7683 ± 0.0257 | 0.0265 ± 0.0026 | 0.0675 ± 0.0212 |
| Teacher-temperature sharpening | 3.7725 ± 0.0243 | 0.0240 ± 0.0021 | 0.0589 ± 0.0165 |

## Predeclared Criteria

| Criterion | Result | Assessment |
|---|---:|---|
| Exact alpha-zero preservation, alpha-one endpoint, and final-slice isolation | 5/5 seeds; max alpha-zero deviation 0.00e+00 | **Pass** |
| Self sharpening reduces mean entropy versus CE | 0.0324 | **Pass** |
| Teacher sharpening reduces mean entropy versus CE | 0.0283 | **Pass** |
| Teacher sharpening wins versus CE | 5/5; mean advantage 0.0133 | **Fail** |
| Teacher sharpening wins versus self sharpening | 5/5; mean advantage 0.0086 | **Fail** |
| Teacher sharpening fresh ECE increase versus CE <= 0.01 | mean -0.0000; maximum 0.0023 | **Pass** |
| Teacher sharpening within 0.15 loss of frozen teacher | maximum gap 0.0831 | **Pass** |
| Manually reviewed fixed continuations coherent | 5/5 seeds | **Pass** |
| Third layer, scaling, or quantization authorization | — | **Denied** |

## Fixed Scientific-Prompt Continuations

### Seed 20260871

| Condition | Continuation |
|---|---|
| CE continuation | The purpose of scientific research is to find out the causes of disease and to develop new treatments.<br>The term " |
| Self-entropy sharpening | The purpose of scientific research is to find out the causes of disease and to develop new treatments.<br>The term " |
| Teacher-temperature sharpening | The purpose of scientific research is to find out the causes of disease and to develop new treatments.<br>The term " |

### Seed 20260872

| Condition | Continuation |
|---|---|
| CE continuation | The purpose of scientific research is to find out the causes of disease and to develop new drugs and treatments.<br>The |
| Self-entropy sharpening | The purpose of scientific research is to find out the causes of disease and to develop new drugs and treatments.<br>The |
| Teacher-temperature sharpening | The purpose of scientific research is to find out the causes of disease and to develop new drugs and treatments.<br>The |

### Seed 20260873

| Condition | Continuation |
|---|---|
| CE continuation | The purpose of scientific research is to find out the causes of disease and to develop new drugs to treat the disease. |
| Self-entropy sharpening | The purpose of scientific research is to find and develop new and improved methods of treatment and prevention of disease.<br>The |
| Teacher-temperature sharpening | The purpose of scientific research is to find out the causes of disease and to develop new drugs to treat the disease. |

### Seed 20260874

| Condition | Continuation |
|---|---|
| CE continuation | The purpose of scientific research is to understand the world and to make it better.<br>The scientific research is a very |
| Self-entropy sharpening | The purpose of scientific research is to understand the world and to make it better.<br>The scientific research is a very |
| Teacher-temperature sharpening | The purpose of scientific research is to understand the world and to make it better.<br>The purpose of scientific research is |

### Seed 20260875

| Condition | Continuation |
|---|---|
| CE continuation | The purpose of scientific research is to find out the truth about the world.<br>The scientific method is a systematic approach |
| Self-entropy sharpening | The purpose of scientific research is to provide a better understanding of the world.<br>The scientific method is a systematic approach |
| Teacher-temperature sharpening | The purpose of scientific research is to provide a better understanding of the world.<br>The scientific method is a systematic approach |
