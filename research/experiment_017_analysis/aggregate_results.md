# Experiment 017: Fresh-Test Aggregate Results

## Fresh Held-Out Test Loss

| Seed | Frozen teacher | CE-calibrator | Value-functional | Moment-functional | CE − moment | Value − moment |
|---:|---:|---:|---:|---:|---:|---:|
| 20260851 | 3.9639 | 4.0114 | 4.0432 | 4.4284 | -0.4170 | -0.3852 |
| 20260852 | 3.9639 | 4.0041 | 4.0553 | 4.4236 | -0.4194 | -0.3683 |
| 20260853 | 3.9639 | 3.9939 | 4.0484 | 4.4398 | -0.4459 | -0.3914 |
| 20260854 | 3.9639 | 4.0196 | 4.0349 | 4.3893 | -0.3697 | -0.3545 |
| 20260855 | 3.9639 | 4.0155 | 4.0334 | 4.4055 | -0.3900 | -0.3722 |

## Aggregate

| Metric | Mean ± sample SD |
|---|---:|
| teacher fresh test loss | 3.9639 ± 0.0000 |
| ce calibrator loss | 4.0089 ± 0.0102 |
| value functional loss | 4.0430 ± 0.0092 |
| moment functional loss | 4.4173 ± 0.0199 |
| ce minus value | -0.0341 ± 0.0182 |
| ce minus moment | -0.4084 ± 0.0293 |
| value minus moment | -0.3743 ± 0.0145 |
| ce calibrator l1 log var | 1.2034 ± 0.0917 |
| value functional l1 log var | 1.2757 ± 0.0836 |
| moment functional l1 log var | 0.1877 ± 0.0072 |
| ce calibrator l1 value nmse | 1.0949 ± 0.0404 |
| value functional l1 value nmse | 0.1479 ± 0.0019 |
| moment functional l1 value nmse | 0.3148 ± 0.0116 |

## Acceptance

| Rule | Outcome |
|---|---|
| alpha zero exact all conditions all seeds | True |
| all complete two layer endpoints | True |
| moment beats ce count | 0 |
| moment beats value count | 0 |
| moment beats ce mean by at least 0 05 | False |
| moment beats value mean by at least 0 03 | False |
| moment improves l1 log variance over value count | 5 |
| moment within 0 15 of teacher | False |
| third layer permitted | False |

All final metrics above come from the fresh test slice of eligible WikiText-2 test sequences 129–256. Each seed trained all three conditions before that slice was requested for evaluation.
