# Phase 2: Swap Engine — LLM Paradigm

**Document:** PARADOM-PHASE-002
**Duration:** Months 4–6
**Prerequisites:** Phase 1 complete. Core experiment shows positive signal (Z < Y + 30%).
**Goal:** Build a production-quality swap engine for the full LLM paradigm — all major conversion paths.

---

## Phase 2 Objective

> Expand from the Phase 1 toy-model proof of concept to a fully functional LLM swap engine that handles real 7B–70B models, all major LLM architectures, and all three major LLM conversion paths.

---

## Success Criteria

- [ ] `paradom swap` handles: Transformer→Mamba, Transformer→MoE, Transformer→Transformer
- [ ] Tested on: LLaMA 3 8B, Mistral 7B, Falcon 7B as sources
- [ ] Transformer→Mamba achieves ≥ 60% quality retention on WikiText-2
- [ ] Transformer→MoE achieves ≥ 70% quality retention
- [ ] 70B model support confirmed (streaming mode, <16GB RAM)
- [ ] `paradom identify` shows equivalence maps for all 3 conversion paths
- [ ] Benchmark report published: public, honest quality numbers

---

## Month 4: Transformer → Mamba (Full Scale)

### Scale Up from Phase 1 Toy Model

Phase 1 proved the concept on 10M param models. Phase 2 scales to real models.

**Key challenges at 7B scale:**
- SVD on 4096×4096 matrices: slow without optimization
- Mamba config must match LLaMA depth exactly (32 layers → 32 SSM layers)
- State dimension (N) choice significantly affects quality

**Optimization tasks:**
```python
# Randomized SVD for large matrices (10-50× faster than exact SVD)
def fast_svd(W: Tensor, rank: int = None, n_iter: int = 4) -> SVDFactors:
    """
    Randomized SVD via sklearn.utils.extmath.randomized_svd
    Near-identical quality to exact SVD for top-k components.
    Scales to 8192×8192 matrices in seconds.
    """
    from sklearn.utils.extmath import randomized_svd
    U, S, Vh = randomized_svd(
        W.numpy(), 
        n_components=rank or min(W.shape),
        n_iter=n_iter,
        random_state=42
    )
    return SVDFactors(torch.tensor(U), torch.tensor(S), torch.tensor(Vh))
```

**State dimension sensitivity study:**
```
Run LLaMA 3 8B → Mamba with state_dim = [16, 32, 64, 128, 256]
Measure perplexity for each. Find optimal N.
Document in: research/STATE_DIM_SENSITIVITY.md
```

---

## Month 5: Transformer → MoE & Mapping Registry

### Dense → MoE Expert Initialization

```python
class DenseToMoEMapper:
    """
    Converts a dense FFN to a Mixture of Experts layer.
    
    Core insight:
      A dense FFN weight matrix can be decomposed into N "expert subspaces"
      via SVD. Each expert specializes in a subspace of the original function.
      The router is initialized uniformly — experts differentiate during use.
    """
    
    def map_ffn_to_moe(
        self,
        W_up: Tensor,      # Dense FFN up-projection: (d_ff, d_model)
        W_down: Tensor,    # Dense FFN down-projection: (d_model, d_ff)
        n_experts: int,
        top_k: int = 2     # How many experts are active per token
    ) -> MoELayer:
        
        # SVD decompose the up projection
        U, S, Vh = torch.linalg.svd(W_up, full_matrices=False)
        
        # Divide singular value spectrum evenly across experts
        expert_rank = len(S) // n_experts
        experts = []
        
        for i in range(n_experts):
            start, end = i * expert_rank, (i+1) * expert_rank
            
            # Expert i specializes in singular dimensions [start:end]
            # These dimensions represent a "sub-function" of the original FFN
            W_up_expert   = U[:, start:end] * S[start:end]
            W_down_expert = Vh[start:end, :].T  # Note: transpose for down proj
            
            experts.append(ExpertFFN(W_up=W_up_expert, W_down=W_down_expert))
        
        # Router: uniform initialization (all experts equally weighted initially)
        router_weight = torch.zeros(n_experts, W_up.shape[1])
        nn.init.xavier_uniform_(router_weight)
        
        return MoELayer(experts=experts, router=router_weight, top_k=top_k)
```

### Mapping Registry

```python
# All supported LLM conversion paths in Phase 2
MAPPING_REGISTRY = {
    # LLM paradigm
    ("transformer", "mamba"):       TransformerToMambaMapper,
    ("transformer", "moe"):         TransformerToMoEMapper,
    ("transformer", "transformer"): TransformerToTransformerMapper,
    ("llama",   "mistral"):         LlamaToMistralMapper,    # Near-identical arch
    ("llama",   "mamba"):           TransformerToMambaMapper,
    ("mistral", "mamba"):           TransformerToMambaMapper,
    ("falcon",  "mamba"):           TransformerToMambaMapper,
}
```

---

## Month 6: Quality Hardening & Public Benchmark

### Benchmark Protocol

Every converted model must pass through the standard benchmark before release:

```bash
# Standard Phase 2 benchmark run
paradom validate \
  --source meta-llama/Llama-3-8B \
  --swapped ./output/llama3_as_mamba \
  --benchmark perplexity,hellaswag,arc_easy \
  --report ./benchmarks/phase2_llama3_to_mamba.json
```

### Phase 2 Quality Targets

| Conversion | Perplexity Retention | HellaSwag Retention | Target |
|---|---|---|---|
| LLaMA 3 8B → Transformer (same-ish) | ≥ 82% | ≥ 80% | Phase 2 |
| LLaMA 3 8B → Mamba 7B | ≥ 60% | ≥ 55% | Phase 2 |
| LLaMA 3 8B → MoE 7B (8 experts) | ≥ 70% | ≥ 65% | Phase 2 |
| Mistral 7B → Mamba 7B | ≥ 58% | ≥ 53% | Phase 2 |

### Publish Results Openly

Phase 2 ends with publishing honest benchmark results to:
- The project GitHub repository
- A technical blog post: "Paradom Phase 2: What We Learned About Weight Swapping at Scale"

This includes results that are worse than expected, if that is what we find.

---

## Phase 2 Exit Criteria

1. ✅ All 3 LLM conversion paths working
2. ✅ Quality targets met (or revised with explanation if not)
3. ✅ 70B model processed in <16GB RAM
4. ✅ Public benchmark results published
5. ✅ Phase 3 plan finalized based on Phase 2 quality findings
