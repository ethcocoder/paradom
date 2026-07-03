Experiment 001: Transformer → Mamba Weight Equivalence
Phase 1 core proof using strengthened explicit mapping + SSM derivation.

Perplexity Results
Model	Perplexity
Transformer (Source)	17.21
Mamba (Trained Baseline)	23.86
Mamba (Random Init)	59318.80
Swapped Mamba (Paradom)	67782.72
Metrics
Swap fraction: 20%
Intelligence retention: -14.3%
Mean CKA: 0.167
Quality tier: degraded
Layers converted: 21
Swap types: {'direct': 0.23809523809523808, 'projected': 0.38095238095238093, 'derived': 0.2857142857142857}
Phase 1 Verdict
FAILURE — Swapped model does NOT beat random initialization. Swap failed to preserve meaningful intelligence. Source Transformer (17 ppl) remains stronger than swapped target (67783 ppl), as expected for cross-arch transfer.

Mapping Used
Transformer	Mamba	Method
q_proj + k_proj	in_proj (1st half)	Direct concat
v_proj	in_proj (2nd half)	Direct concat
o_proj	out_proj	SVD projected
gate + up	x_proj	SVD projected
down	dt_proj	SVD projected
input_layernorm	norm	Direct
Q @ K^T eigenstructure	A_log	Derived
v_proj energy	D, conv1d	Derived
embedding, lm_head	embedding, lm_head	Direct
