# Experiment 019: Output-Sharpening Source Notes

**Purpose:** These notes distinguish a legitimate end-to-end output objective from a claim that reduced token entropy alone proves successful Transformer-to-Mamba knowledge transfer.

## Verified Findings

| Source | Finding relevant to the design | Design consequence |
|---|---|---|
| Carlsson et al., ICLR 2025, *The Hyperfitting Phenomenon* [1] | Fine-tuning an autoregressive model to near-zero loss on a small sample set can yield very low-entropy output distributions and was reported to improve long-form greedy generation in that study. | A low-entropy output can be an empirical behavior worth measuring, but it is not evidence of cross-architecture transfer and must be compared with ordinary CE continuation. |
| Huang et al., ICLR 2025, *Self-Improvement in Language Models: The Sharpening Mechanism* [2] | The paper formalizes sharpening as concentrating probability mass on high-quality sequences. Its analysis states that SFT sharpening needs coverage under the initial policy. | A teacher-sharpened objective must retain CE supervision and use the frozen teacher only as a declared training signal; it cannot be interpreted as creating new knowledge or as standalone Mamba conversion. |
| Pereyra et al., 2017, *Regularizing Neural Networks by Penalizing Confident Output Distributions* [3] | Confidence penalties and label smoothing modify output entropy and have broad supervised-learning effects; entropy is an objective property, not a direct language-quality proof. | Add output entropy and calibration diagnostics but retain next-token loss as the primary endpoint; never select a condition merely because it is sharper. |
| Xie et al., EMNLP 2024, *Calibrating Language Models with Adaptive Temperature Scaling* [4] | Token-level temperatures can change calibration, and post-training can alter the relationship between confidence and correctness. | Report token entropy and an expected-calibration-error-style fixed-bin diagnostic. A sharper distribution that worsens held-out loss or calibration is a failure, even if prompts look fluent. |
| Kapoor et al., Uncertainty in NLP 2024, *Calibration-Tuning* [5] | Fine-tuning can improve calibration without affecting accuracy, emphasizing that confidence and accuracy are distinct outcomes. | Separate quality metrics (held-out NLL/CE), transfer metrics (teacher-sharpened condition versus equal CE/sharpening controls), and confidence metrics. |

## Design Constraint Derived from the Literature

Experiment 019 must not optimize a local Mamba-to-attention target. It should instead use a two-stage, fully deployed hybrid: first establish a two-layer alpha-one endpoint using the prior conditional CE transport architecture; then apply a matched short output-level fine-tuning stage. The final loss must compare:

1. **ordinary CE continuation** to quantify ordinary extra adaptation;
2. **self-entropy sharpening** to quantify non-teacher output concentration; and
3. **teacher-temperature sharpening** to quantify a teacher-specific output signal.

The teacher-specific condition must beat both controls on a fresh held-out loss slice to support a narrow knowledge-transfer claim. Entropy reduction alone is neither a primary endpoint nor a success criterion.

## References

[1] [Carlsson et al. *The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation.* ICLR 2025.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/acfb8a98d2af3b3b8d5a9ffa979954a7-Abstract-Conference.html)

[2] [Huang et al. *Self-Improvement in Language Models: The Sharpening Mechanism.* ICLR 2025.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bee8c2bc757f6bbc3efd7cf1b979f0c9-Abstract-Conference.html)

[3] [Pereyra et al. *Regularizing Neural Networks by Penalizing Confident Output Distributions.* 2017.](https://arxiv.org/abs/1701.06548)

[4] [Xie et al. *Calibrating Language Models with Adaptive Temperature Scaling.* EMNLP 2024.](https://aclanthology.org/2024.emnlp-main.1007/)

[5] [Kapoor et al. *Calibration-Tuning: Teaching Large Language Models to Know What They Don't Know.* Uncertainty in NLP 2024.](https://aclanthology.org/2024.uncertainlp-1.1/)
