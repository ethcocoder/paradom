# Technical Specifications: Paradom Framework

**Document:** PARADOM-SPEC-001
**Version:** 2.0.0
**Date:** 2026-06-30

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Weight Equivalence Specification](#2-weight-equivalence-specification)
3. [Swap Operation Specifications](#3-swap-operation-specifications)
4. [Importance Scoring Specifications](#4-importance-scoring-specifications)
5. [Paradigm Mapping Specifications](#5-paradigm-mapping-specifications)
6. [Validation Specifications](#6-validation-specifications)
7. [Configuration Specification](#7-configuration-specification)
8. [Error Handling Specification](#8-error-handling-specification)
9. [Performance Specifications](#9-performance-specifications)

---

## 1. System Requirements

### 1.1 Runtime Requirements (Swap Mode)

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| Python | 3.10+ | 3.11+ | |
| RAM | 4GB | 16GB | Streaming mode keeps peak ~1-2GB per layer |
| CPU Cores | 2 | 8+ | More cores = faster parallel layer processing |
| GPU | Not required | Optional (A10/3090) | Only needed for calibration pass |
| Storage | 2× model size | 3× model size | Input + output + temp |
| OS | Linux / macOS | Ubuntu 22.04 LTS | |

### 1.2 Python Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.1.0"
transformers = "^4.40.0"
safetensors = "^0.4.0"
numpy = "^1.26.0"
scipy = "^1.12.0"          # SVD, eigendecomposition, Procrustes
scikit-learn = "^1.4.0"    # PCA, CKA computation
POT = "^0.9.0"             # Python Optimal Transport
huggingface-hub = "^0.22.0"
gymnasium = "^0.29.0"      # RL environment support
pyyaml = "^6.0.1"
typer = "^0.12.0"
rich = "^13.7.0"
tqdm = "^4.66.0"
psutil = "^5.9.0"

[tool.poetry.dev-dependencies]
pytest = "^8.0"
pytest-benchmark = "^4.0"
lm-eval = "^0.4.0"
```

---

## 2. Weight Equivalence Specification

### 2.1 Functional Role Taxonomy

Every weight in every paradigm is assigned exactly one functional role from this universal taxonomy:

```python
class FunctionalRole(Enum):
    # Universal roles (appear in ALL paradigms)
    EMBEDDING          = "embedding"           # Input representation
    OUTPUT_HEAD        = "output_head"         # Final prediction layer
    NORMALIZATION      = "normalization"       # Scale/shift parameters
    BIAS               = "bias"               # Additive offset terms

    # Context & attention roles (LLM, ViT, GNN, Multimodal)
    CONTEXT_QUERY      = "context_query"       # Q matrices, SSM B proj
    CONTEXT_KEY        = "context_key"         # K matrices, SSM C proj
    CONTEXT_VALUE      = "context_value"       # V matrices, value aggregation
    CONTEXT_OUTPUT     = "context_output"      # Output projection

    # Feed-forward roles (LLM, Vision, RL)
    FFN_EXPAND         = "ffn_expand"          # Up/gate projections
    FFN_CONTRACT       = "ffn_contract"        # Down projections

    # Vision-specific roles
    SPATIAL_FILTER     = "spatial_filter"      # CNN convolutional filters
    PATCH_EMBED        = "patch_embed"         # ViT patch embedding

    # RL-specific roles
    STATE_ENCODER      = "state_encoder"       # Feature extraction layers
    ACTION_HEAD        = "action_head"         # Policy output layers
    VALUE_HEAD         = "value_head"          # Value estimation layers

    # Generative-specific roles
    NOISE_PREDICTOR    = "noise_predictor"     # Diffusion UNet core
    FLOW_PREDICTOR     = "flow_predictor"      # Flow matching vector field
    TIME_EMBED         = "time_embed"          # Timestep conditioning

    # Graph-specific roles
    NODE_TRANSFORM     = "node_transform"      # Node feature transformation
    EDGE_TRANSFORM     = "edge_transform"      # Edge feature transformation
    AGGREGATION        = "aggregation"         # Message aggregation weights
```

### 2.2 Equivalence Score Computation

```python
@dataclass
class EquivalenceScore:
    source_layer: str           # e.g. "model.layers.0.self_attn.q_proj"
    target_layer: str           # e.g. "backbone.layers.0.mixer.in_proj"
    functional_role_match: bool # Same FunctionalRole enum value
    cka_score: float            # [0, 1] — 1.0 = identical representations
    dimension_match: bool       # Source and target have same shape
    swap_type: str              # "direct" | "projected" | "tensor" | "ot"
    confidence: float           # Overall confidence in this equivalence
    
    @property
    def is_safe_to_swap(self) -> bool:
        return (
            self.functional_role_match and
            self.cka_score > 0.40 and
            self.confidence > 0.60
        )
```

### 2.3 Equivalence Confidence Thresholds

```python
CONFIDENCE_THRESHOLDS = {
    "direct_swap":     {"cka_min": 0.70, "role_match": True},   # Direct copy
    "projected_swap":  {"cka_min": 0.40, "role_match": True},   # With projection
    "tensor_swap":     {"cka_min": 0.35, "role_match": True},   # Tucker decomp
    "ot_swap":         {"cka_min": 0.20, "role_match": True},   # Optimal transport
    "skip":            {"cka_min": 0.00, "role_match": False},  # Initialize from scratch
}
```

---

## 3. Swap Operation Specifications

### 3.1 Direct Swap

**When used:** Source and target weight have identical shape AND CKA > 0.70

```python
def direct_swap(W_source: Tensor, W_target_shape: tuple) -> Tensor:
    """
    Direct copy of source weight to target position.
    No mathematical transformation applied.
    Preserves: All information in source weight.
    Quality: Highest possible (bounded by functional equivalence).
    """
    assert W_source.shape == W_target_shape, \
        f"Direct swap requires identical shapes: {W_source.shape} vs {W_target_shape}"
    return W_source.clone().detach()
```

**Expected quality retention:** 85–99%
**Compute cost:** Negligible (memory copy only)
**RAM overhead:** 1× layer size

---

### 3.2 Projected Swap

**When used:** Dimensions differ (hidden_size_source ≠ hidden_size_target), CKA > 0.40

```python
def projected_swap(
    W_source: Tensor,        # Shape: (d_out_src, d_in_src)
    target_shape: tuple,     # (d_out_tgt, d_in_tgt)
    method: str = "procrustes"
) -> Tensor:
    """
    Projects source weight into target dimensionality.
    Preserves maximum variance directions (most important numbers).
    
    Methods:
      procrustes: Optimal rotation + scaling (best for same-scale dims)
      pca:        PCA projection (best for large → small)
      pad:        Zero-pad + scale (best for small → large)
    """
    d_out_tgt, d_in_tgt = target_shape
    
    if method == "procrustes":
        # SVD of source
        U, S, Vh = torch.linalg.svd(W_source, full_matrices=False)
        
        # Truncate or pad singular dimensions
        r = min(len(S), d_out_tgt, d_in_tgt)
        U_r  = U[:d_out_tgt, :r]    # Truncate to target output dim
        S_r  = S[:r]
        Vh_r = Vh[:r, :d_in_tgt]   # Truncate to target input dim
        
        # Reconstruct in target shape
        W_projected = (U_r * S_r.unsqueeze(0)) @ Vh_r
        return W_projected
        
    elif method == "pca":
        # Keep top principal components (maximum variance = most intelligence)
        ...
    
    elif method == "pad":
        # Expand small → large via padding + noise
        padding_std = W_source.std() * 0.01  # 1% noise initialization
        ...
```

**Expected quality retention:** 70–90%
**Compute cost:** Low (one SVD per layer)
**RAM overhead:** 2× layer size

---

### 3.3 Tensor Decomposition Swap

**When used:** Multi-dimensional tensors (CNN 4D filters, 3D tensors), CKA > 0.35

```python
def tensor_decomposition_swap(
    W_source: Tensor,     # Shape: (C_out, C_in, H, W) for CNN
    target_shape: tuple,  # Target filter shape
    rank_fraction: float = 0.75
) -> Tensor:
    """
    Tucker decomposition to extract core tensor (the essential learned pattern),
    then recompose in the target architecture's tensor structure.
    
    This handles the fundamental shape mismatch between CNN filters and ViT patches.
    
    Example:
      CNN filter:  (256, 128, 3, 3) — 256 output channels, 3×3 spatial
      ViT weight:  (768, 768)       — patch embedding dimension
      
      Step 1: Tucker decompose CNN filter → core tensor G + factor matrices
      Step 2: G encodes the "essential spatial pattern" independent of shape
      Step 3: Recompose G into ViT weight shape with new factor matrices
    """
    from torch_tucker import tucker_decompose, tucker_reconstruct
    
    # Decompose
    ranks = [int(d * rank_fraction) for d in W_source.shape]
    G, factors = tucker_decompose(W_source, ranks=ranks)
    
    # Recompose into target shape
    target_factors = [_project_factor(f, t) for f, t in zip(factors, target_shape)]
    W_target = tucker_reconstruct(G, target_factors)
    
    return W_target
```

**Expected quality retention:** 60–80%
**Compute cost:** Medium (Tucker decomposition)
**RAM overhead:** 2-3× layer size

---

### 3.4 Optimal Transport Swap

**When used:** Low CKA (0.20–0.40), structurally different paradigms (e.g., GNN → Transformer)

```python
def optimal_transport_swap(
    W_source: Tensor,
    W_target_init: Tensor,   # Target architecture's default initialization
    reg: float = 0.1         # Entropic regularization
) -> Tensor:
    """
    Uses Wasserstein optimal transport to find the mapping between
    source weight distribution and target weight distribution that
    preserves the most structural information.
    
    Think of it as: "Move the source weights to the target space
    in the most faithful way possible, given the constraints."
    
    Used for cross-paradigm swaps where direct structural matching fails.
    """
    import ot  # Python Optimal Transport library
    
    # Flatten to 1D distributions
    src_dist = W_source.flatten().numpy()
    tgt_shape = W_target_init.shape
    tgt_dist = W_target_init.flatten().numpy()
    
    # Compute cost matrix (Euclidean distance between weight values)
    n_src, n_tgt = len(src_dist), len(tgt_dist)
    M = ot.dist(src_dist.reshape(-1,1), tgt_dist.reshape(-1,1))
    M /= M.max()  # Normalize
    
    # Solve optimal transport with Sinkhorn (efficient, regularized)
    T = ot.sinkhorn(
        a=np.ones(n_src)/n_src,
        b=np.ones(n_tgt)/n_tgt,
        M=M,
        reg=reg
    )
    
    # Apply transport map
    W_transported = (T @ src_dist.reshape(-1, 1)).reshape(tgt_shape)
    return torch.tensor(W_transported, dtype=W_source.dtype)
```

**Expected quality retention:** 45–65%
**Compute cost:** High (OT scales quadratically)
**RAM overhead:** 4× layer size

---

## 4. Importance Scoring Specifications

### 4.1 Methods

```python
class ImportanceMethods:
    
    @staticmethod
    def svd_spectrum(
        W: Tensor,
        top_k_fraction: float = 0.20
    ) -> Tensor:
        """
        Returns a mask of the top-k% weights by their contribution
        to the singular value spectrum.
        
        Rationale: Weights that contribute to top singular values
        carry the most variance = most information = most intelligence.
        
        Fast, no data needed, mathematically grounded.
        Recommended for: LLMs, Vision models
        """
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # Cumulative energy: how much of total variance is in top-k values
        energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
        top_k = (energy < top_k_fraction).sum() + 1
        
        # Reconstruct importance scores for each weight element
        importance = (U[:, :top_k].abs().sum(1).unsqueeze(1) *
                     Vh[:top_k, :].abs().sum(0).unsqueeze(0))
        return importance

    @staticmethod
    def gradient_sensitivity(
        model,
        representative_inputs: Tensor,
        top_k_fraction: float = 0.20
    ) -> Dict[str, Tensor]:
        """
        Computes gradient magnitude of loss with respect to each weight.
        High gradient = high sensitivity = important weight.
        
        Requires: Small set of representative inputs (~100 samples)
        No labels needed (use self-supervised gradient signal)
        """
        ...

    @staticmethod  
    def activation_magnitude(
        model,
        representative_inputs: Tensor,
        top_k_fraction: float = 0.20
    ) -> Dict[str, Tensor]:
        """
        Weights connected to high-magnitude activations are most "active"
        in the forward pass. High activity = high importance.
        """
        ...

    @staticmethod
    def lottery_ticket(
        W: Tensor,
        sparsity: float = 0.80
    ) -> Tensor:
        """
        Classic magnitude-based pruning mask.
        Keep the top (1-sparsity)% weights by absolute magnitude.
        Based directly on the Lottery Ticket Hypothesis.
        """
        threshold = W.abs().flatten().quantile(sparsity)
        return (W.abs() >= threshold).float()
```

### 4.2 Recommended Methods by Paradigm

| Paradigm | Recommended Method | Swap Fraction | Rationale |
|---|---|---|---|
| LLM (Transformer) | `svd_spectrum` | 15–25% | No data needed, fast on large matrices |
| LLM (Mamba/MoE) | `svd_spectrum` | 15–25% | Same rationale |
| Vision (CNN) | `activation_magnitude` | 20–30% | Spatial filters benefit from activation info |
| Vision (ViT) | `svd_spectrum` | 15–25% | Attention-like structure, same as LLM |
| RL | `gradient_sensitivity` | 25–35% | RL weights are more heterogeneous |
| Diffusion | `svd_spectrum` | 20–30% | UNet structure similar to vision |
| GNN | `activation_magnitude` | 25–35% | Graph structure benefits from activation patterns |

---

## 5. Paradigm Mapping Specifications

### 5.1 LLM Paradigm: Transformer → Mamba

**Layer correspondence table:**

| Transformer Layer | Mamba Layer | Swap Type | Expected CKA |
|---|---|---|---|
| `q_proj` + `k_proj` (combined) | `in_proj` (first half) | Projected | 0.55–0.75 |
| `v_proj` | `in_proj` (second half) | Projected | 0.60–0.80 |
| `o_proj` | `out_proj` | Direct/Projected | 0.65–0.85 |
| `gate_proj` + `up_proj` | `x_proj` | Projected | 0.50–0.70 |
| `down_proj` | `dt_proj` | Projected | 0.45–0.65 |
| `input_layernorm` | `norm` | Direct | 0.90–0.99 |
| `embed_tokens` | `embedding` | Direct | 0.95–0.99 |
| `lm_head` | `lm_head` | Direct | 0.95–0.99 |

**Special handling — A matrix (SSM state dynamics):**
```python
def derive_A_matrix_from_attention(W_Q: Tensor, W_K: Tensor, state_dim: int) -> Tensor:
    """
    The Mamba A matrix has no direct Transformer equivalent.
    It is DERIVED from the attention pattern eigenstructure.
    
    This is not a swap — it is a mathematical derivation.
    The A matrix IS the eigenvalue spectrum of the attention pattern matrix.
    """
    W_QK = W_Q @ W_K.T                              # Attention pattern matrix
    eigenvalues, _ = torch.linalg.eigh(W_QK)        # Eigendecompose
    top_eigenvalues = eigenvalues[-state_dim:]       # Take top N
    A_log = torch.log(top_eigenvalues.abs() + 1e-6) # SSM parameterizes log(-A)
    return A_log
```

---

### 5.2 Vision Paradigm: CNN → ViT

**Layer correspondence table:**

| CNN Component | ViT Component | Swap Type | Notes |
|---|---|---|---|
| Early conv layers (1–3) | Patch embedding | Tensor decomp | Spatial feature → patch feature |
| Mid conv layers (4–6) | Attention Q/K/V | Tensor decomp | Feature detection → attention |
| Late conv layers (7+) | MLP layers | Projected | High-level features → FFN |
| BatchNorm | LayerNorm | Direct (scale only) | Normalize weight, drop running stats |
| FC classifier | MLP head | Direct/Projected | Classification weights |

---

### 5.3 RL Paradigm: DQN → PPO

**Layer correspondence table:**

| DQN Component | PPO Component | Swap Type | Swappable? |
|---|---|---|---|
| Feature extractor (conv/MLP) | Policy feature extractor | Direct/Projected | ✅ Yes — highest value swap |
| Q-value head (linear) | Action logits head | Projected + sign transform | ⚠️ Partial |
| — | Value head | Xavier init | ❌ No equivalent — initialize fresh |

**Key insight:** The DQN feature extractor and PPO policy backbone encode the same concept — "how to represent a state." These are the weights worth swapping. The Q-value → policy head translation requires a softmax transformation.

---

### 5.4 Generative Paradigm: Diffusion → Flow Matching

**Layer correspondence table:**

| Diffusion Component | Flow Matching Component | Swap Type | Notes |
|---|---|---|---|
| UNet encoder blocks | Flow encoder blocks | Direct/Projected | Visual feature extraction |
| UNet decoder blocks | Flow decoder blocks | Direct/Projected | Feature upsampling |
| Time embedding | Time embedding | Direct | Identical functional role |
| Cross-attention (text) | Cross-attention (text) | Direct | Identical architecture |
| Output conv (noise pred) | Output conv (flow pred) | Direct + sign flip | Flow = -noise direction |

**Mathematical basis for output sign flip:**
```
Diffusion output: ε_θ(x_t, t)  — predicted noise (subtract from x_t)
Flow output:      v_θ(x_t, t)  — predicted velocity (direction toward data)

Relationship: v_θ ≈ -ε_θ  (approximately, under specific parameterizations)

Paradom handles this with: W_flow_out = -W_diffusion_out
```

---

## 6. Validation Specifications

### 6.1 Universal Metrics (All Paradigms)

```python
@dataclass
class SwapValidationReport:
    # Identity
    source_model: str
    target_architecture: str
    source_paradigm: str
    target_paradigm: str
    
    # Swap statistics
    total_weights: int
    weights_swapped: int
    swap_fraction: float             # weights_swapped / total_weights
    swap_type_distribution: Dict     # {"direct": 0.4, "projected": 0.5, ...}
    
    # Quality metrics
    cka_scores: Dict[str, float]     # Per-layer CKA score
    mean_cka: float                  # Average CKA across all swapped layers
    
    # Paradigm-specific quality
    paradigm_metric_name: str        # e.g. "perplexity" for LLM, "top1_accuracy" for vision
    source_paradigm_metric: float    # Baseline (source model quality)
    converted_paradigm_metric: float # After swap quality
    retention_fraction: float        # converted/source (higher is better)
    
    # Quality tier
    quality_tier: str                # "excellent" | "good" | "acceptable" | "degraded"
    recommendation: str
    
    # Resource usage
    conversion_time_seconds: float
    peak_ram_mb: float
    peak_vram_mb: float              # 0 if CPU-only
```

### 6.2 Paradigm-Specific Quality Metrics

| Paradigm | Primary Metric | Secondary Metric | Tool |
|---|---|---|---|
| LLM | Perplexity ratio | HellaSwag, MMLU accuracy | lm-eval-harness |
| Vision | Top-1 accuracy delta | CKA layer similarity | torchmetrics |
| RL | Episode reward ratio | Policy entropy | gymnasium |
| Diffusion → Flow | FID score delta | CLIP score | torch-fidelity |
| GNN | Node classification accuracy | Link prediction AUC | PyG benchmarks |

### 6.3 Quality Tiers (Universal)

```python
QUALITY_TIERS = {
    "excellent":   {"retention": (0.85, 1.00), "mean_cka": (0.70, 1.00)},
    "good":        {"retention": (0.70, 0.85), "mean_cka": (0.55, 0.70)},
    "acceptable":  {"retention": (0.55, 0.70), "mean_cka": (0.40, 0.55)},
    "degraded":    {"retention": (0.00, 0.55), "mean_cka": (0.00, 0.40)},
}
```

---

## 7. Configuration Specification

```yaml
# paradom_config.yaml

swap:
  importance_method: "svd_spectrum"   # svd_spectrum | gradient_sensitivity | activation_magnitude | lottery_ticket
  swap_fraction: 0.20                 # Fraction of weights to swap (top k% by importance)
  swap_type: "auto"                   # auto | direct | projected | tensor | ot
  streaming: true                     # Layer-by-layer (low RAM) vs full model (fast)
  max_ram_gb: 8                       # Cap RAM usage (streaming will enforce this)
  parallel_layers: 4                  # How many layers to process in parallel

equivalence:
  cka_threshold_direct: 0.70         # Min CKA for direct swap
  cka_threshold_projected: 0.40      # Min CKA for projected swap
  cka_threshold_tensor: 0.35         # Min CKA for tensor swap
  cka_threshold_ot: 0.20             # Min CKA for OT swap
  cka_sample_size: 256               # Samples for CKA computation

output:
  format: "safetensors"              # safetensors | pytorch | gguf
  save_report: true
  report_format: "json"
  save_equivalence_map: true         # Save which weights were swapped where

validation:
  run_after_swap: true
  paradigm_benchmark: true           # Run paradigm-specific benchmark
  cka_validation: true               # Validate layer representations
  fail_below_tier: "degraded"

logging:
  level: "INFO"
  progress_bars: true
  log_per_layer: false               # Verbose layer-by-layer logging
```

---

## 8. Error Handling Specification

```python
class ParadomException(Exception): pass

class UnsupportedParadigmError(ParadomException):
    """Source or target paradigm not supported."""

class UnsupportedMappingError(ParadomException):
    """No mapping exists for source_arch → target_arch pair."""

class EquivalenceNotFoundError(ParadomException):
    """CKA score too low for any swap type — layer has no equivalent."""
    def __init__(self, layer_name: str, cka_score: float):
        self.layer_name = layer_name
        self.cka_score = cka_score
        super().__init__(
            f"No equivalent found for layer '{layer_name}' "
            f"(CKA={cka_score:.3f} below minimum threshold). "
            f"This layer will be initialized from scratch."
        )

class ShapeMismatchError(ParadomException):
    """Target shape is incompatible with any projection strategy."""

class QualityBelowThresholdError(ParadomException):
    """Swap quality fell below the configured minimum tier."""
    def __init__(self, tier: str, report: SwapValidationReport):
        self.tier = tier
        self.report = report
```

---

## 9. Performance Specifications

### 9.1 Target Swap Times (Streaming Mode, 8-core CPU)

| Model Size | Same-Arch Swap | Cross-Arch Swap | Cross-Paradigm Swap |
|---|---|---|---|
| 1B params | < 2 min | < 5 min | < 10 min |
| 7B params | < 8 min | < 20 min | < 35 min |
| 13B params | < 15 min | < 35 min | < 60 min |
| 70B params | < 90 min | < 3 hours | < 5 hours |

### 9.2 RAM Usage by Model Size (Streaming Mode)

```
RAM usage in streaming mode is CONSTANT regardless of model size.
Only one layer is in memory at a time.

Peak RAM ≈ max_layer_size × 3  (source layer + target layer + swap workspace)

For 7B model (largest layer ~500MB):
  Peak RAM ≈ 1.5GB

For 70B model (largest layer ~2GB):
  Peak RAM ≈ 6GB

This means: a 16GB laptop can process ANY model size in streaming mode.
```

### 9.3 Quality vs Swap Fraction Trade-off

```
Swap Fraction   | Same-Arch Quality | Cross-Arch Quality | Time Cost
────────────────|───────────────────|────────────────────|──────────
5%              | 70–80%            | 40–55%             | Fastest
10%             | 78–88%            | 50–65%             | Fast
20% (default)   | 82–92%            | 60–75%             | Moderate
35%             | 85–93%            | 65–78%             | Slower
50%             | 86–94%            | 67–80%             | Slow
100%            | 88–95%            | 68–82%             | Slowest

Recommendation: 20% swap fraction gives best quality/time ratio.
Beyond 35%, diminishing returns become significant.
```