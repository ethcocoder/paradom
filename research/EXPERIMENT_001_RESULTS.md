# Experiment 001: Transformer → Mamba Weight Equivalence

Phase 1 core proof using strengthened explicit mapping + SSM derivation.

## Perplexity Results

| Model | Perplexity |
|---|---:|
| Transformer (Source) | 7902.18 |
| Mamba (Trained Baseline) | 31871.23 |
| Mamba (Random Init) | 63929.52 |
| **Swapped Mamba (Paradom)** | **43502.61** |

## Metrics

- **Swap fraction:** 20%
- **Intelligence retention:** 63.7%
- **Mean CKA:** 0.014
- **Quality tier:** degraded
- **Layers converted:** 21
- **Swap types:** {'direct': 0.23809523809523808, 'projected': 0.38095238095238093, 'derived': 0.2857142857142857}

## Phase 1 Verdict

**PARTIAL SUCCESS** — Swapped model beats random initialization. Partial transfer (63.7% retention toward trained baseline) — **hypothesis PARTIALLY SUPPORTED**. Source Transformer (7902 ppl) remains stronger than swapped target (43503 ppl), as expected for cross-arch transfer.

## Mapping Used

| Transformer | Mamba | Method |
|---|---|---|
| q_proj + k_proj | in_proj (1st half) | Direct concat |
| v_proj | in_proj (2nd half) | Direct concat |
| o_proj | out_proj | SVD projected |
| gate + up | x_proj | SVD projected |
| down | dt_proj | SVD projected |
| input_layernorm | norm | Direct |
| Q @ K^T eigenstructure | A_log | Derived |
| v_proj energy | D, conv1d | Derived |
| embedding, lm_head | embedding, lm_head | Direct |
