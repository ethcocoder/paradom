# Technical Specifications: Paradom Framework

**Document:** PARADOM-SPEC-001  
**Version:** 1.0.0  
**Date:** 2026-06-30

---

## 1. System Requirements

### 1.1 Runtime Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.11+ |
| PyTorch | 2.1+ | 2.3+ |
| RAM | 2× model size | 3× model size |
| VRAM (optional) | 0 (CPU-only mode) | 24GB+ for calibration |
| Storage (working) | 3× model size | 5× model size |
| CPU Cores | 4 | 16+ |
| OS | Linux / macOS | Ubuntu 22.04 LTS |

### 1.2 Python Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.1.0"
transformers = "^4.40.0"
safetensors = "^0.4.0"
numpy = "^1.26.0"
scipy = "^1.12.0"        # SVD, eigendecomposition
scikit-learn = "^1.4.0"  # PCA
huggingface-hub = "^0.22.0"
pyyaml = "^6.0.1"
typer = "^0.12.0"        # CLI
rich = "^13.7.0"         # Pretty terminal output
tqdm = "^4.66.0"         # Progress bars
psutil = "^5.9.0"        # Memory monitoring

[tool.poetry.dev-dependencies]
pytest = "^8.0"
pytest-benchmark = "^4.0"
lm-eval = "^0.4.0"       # LM Evaluation Harness
```

---

## 2. Architecture Specification Format

Target architectures are defined in YAML:

```yaml
# configs/example_arch.yaml

name: "MyArch-7B"
type: "transformer"  # transformer | mamba | moe | custom

model:
  vocab_size: 32000
  hidden_size: 4096
  intermediate_size: 11008
  num_hidden_layers: 32
  max_position_embeddings: 4096

attention:
  type: "grouped_query"   # full | grouped_query | multi_query | sliding_window
  num_attention_heads: 32
  num_key_value_heads: 8  # For GQA
  head_dim: 128
  use_rope: true
  rope_theta: 10000.0

ffn:
  type: "gated_silu"      # relu | gelu | gated_silu | gated_gelu
  expansion_ratio: 2.6875

normalization:
  type: "rmsnorm"         # rmsnorm | layernorm
  epsilon: 1.0e-5
  pre_norm: true

embedding:
  tie_word_embeddings: false
  init_std: 0.02

dtype: "bfloat16"         # float32 | float16 | bfloat16
```

---

## 3. Mapping Strategy Specifications

### 3.1 TransformerToTransformerMapper

**Use case:** Same family, different configuration (e.g., LLaMA 7B → LLaMA 13B shape, different head counts)

**Strategy:**

```python
class TransformerToTransformerMapper(BaseMapper):
    """
    Maps between transformer variants.
    Handles: dimension changes, head count changes, GQA/MHA conversions,
             RoPE theta changes, normalization type changes.
    """

    def map_attention(self, source: AttentionGroup) -> AttentionGroup:
        # Case 1: Same dimensions — direct copy
        if source.hidden_size == self.target.hidden_size:
            return source

        # Case 2: Different hidden dim — project via SVD
        if source.hidden_size != self.target.hidden_size:
            return self._project_attention(source)

    def _project_attention(self, source: AttentionGroup) -> AttentionGroup:
        # W_Q: (d_model_src × d_head_src × n_heads) → (d_model_tgt × d_head_tgt × n_heads_tgt)
        # Apply PCA to reshape embedding dimension
        W_Q_projected = self.pca_project(source.W_Q, self.target.hidden_size)
        W_K_projected = self.pca_project(source.W_K, self.target.num_kv_heads * self.target.head_dim)
        W_V_projected = self.pca_project(source.W_V, self.target.num_kv_heads * self.target.head_dim)
        W_O_projected = self.pca_project(source.W_O, self.target.hidden_size)
        return AttentionGroup(W_Q_projected, W_K_projected, W_V_projected, W_O_projected)
```

**Expected quality:** 80–95% retention

---

### 3.2 TransformerToMambaMapper

**Use case:** Converting transformer (LLaMA/Mistral) to Mamba SSM

**This is the most novel mapping in Paradom.**

```python
class TransformerToMambaMapper(BaseMapper):
    """
    Maps transformer attention → Mamba SSM layers.
    
    Theory: Attention computes weighted sums over context.
    SSMs compute exponentially-decaying weighted sums over context.
    The attention weight matrix encodes which context positions matter —
    this maps to the SSM's state decay structure.
    """

    def map_attention(self, source: AttentionGroup) -> MambaBlock:
        # Step 1: Extract attention's "context selection" structure
        W_QK = source.W_Q @ source.W_K.T  # (d_model × d_model)
        eigenvalues, eigenvectors = torch.linalg.eigh(W_QK)
        
        # Step 2: Top eigenvalues → SSM state decay rates
        # Large positive eigenvalues = long-range dependencies
        # Large negative eigenvalues = oscillatory patterns
        top_eigenvalues = eigenvalues[-self.target_state_dim:]
        A_log = torch.log(-top_eigenvalues.abs() + 1e-6)  # SSM parameterizes log(-A)
        
        # Step 3: Eigenvectors × V weights → SSM input/output projections
        B = (eigenvectors[:, -self.target_state_dim:].T @ source.W_V[:self.target_d_inner, :])
        C = (source.W_O[:, :self.target_d_inner] @ eigenvectors[:, -self.target_state_dim:])
        
        # Step 4: Build Mamba block
        return MambaBlock(
            in_proj=self._build_in_proj(source),
            conv1d=self._init_conv1d(),      # Xavier init (no transformer equivalent)
            x_proj=self._build_x_proj(B),
            dt_proj=self._build_dt_proj(source),
            A_log=A_log,
            D=torch.ones(self.target_d_inner),  # Residual connection
            out_proj=self._build_out_proj(C)
        )
```

**Expected quality:** 50–75% retention (research-grade; requires calibration for production)

---

### 3.3 DenseToMoEMapper

**Use case:** Converting a dense FFN to a Mixture of Experts architecture

```python
class DenseToMoEMapper(BaseMapper):
    """
    Maps a dense FFN to an MoE layer.
    
    Strategy: Use SVD to find N "expert subspaces" in the dense FFN weight matrix.
    Each expert specializes in a subspace of the original FFN's function.
    The router is initialized to softly activate all experts equally,
    and experts specialize through calibration.
    """

    def map_ffn(self, source: FFNGroup) -> MoEGroup:
        n_experts = self.target_config.num_experts
        
        # SVD of the up-projection
        U, S, Vh = torch.linalg.svd(source.W_up, full_matrices=False)
        
        # Divide singular value spectrum across experts
        expert_rank = len(S) // n_experts
        experts = []
        
        for i in range(n_experts):
            start = i * expert_rank
            end = start + expert_rank
            
            # Expert i specializes in singular dimensions [start:end]
            W_up_expert = U[:, start:end] * S[start:end].unsqueeze(0)
            W_down_expert = Vh[start:end, :]
            
            experts.append(ExpertFFN(
                W_gate=source.W_gate[:, start:end],   # Gated activation
                W_up=W_up_expert,
                W_down=W_down_expert
            ))
        
        # Router initialized as uniform (all experts equally active initially)
        router = nn.Linear(self.target_config.hidden_size, n_experts, bias=False)
        nn.init.xavier_uniform_(router.weight)
        
        return MoEGroup(experts=experts, router=router)
```

---

### 3.4 Dimension Projection Specifications

When source and target hidden dimensions differ:

```python
class DimensionProjector:
    """
    Handles all cases of embedding dimension mismatch.
    """

    def project(
        self,
        W: Tensor,            # Source weight: (..., d_src, ...)
        src_dim: int,
        tgt_dim: int,
        axis: int = -1        # Which axis to project
    ) -> Tensor:
        
        if src_dim == tgt_dim:
            return W
        
        elif src_dim > tgt_dim:
            # Compression: PCA — keeps maximum variance directions
            return self._pca_compress(W, tgt_dim, axis)
        
        else:
            # Expansion: Pad + initialize new dimensions
            return self._pad_expand(W, tgt_dim, axis)

    def _pca_compress(self, W: Tensor, tgt_dim: int, axis: int) -> Tensor:
        # Move target axis to last position
        W_flat = W.transpose(axis, -1).reshape(-1, W.shape[axis])
        U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
        # Keep top tgt_dim components (maximum variance)
        W_compressed = (U[:, :tgt_dim] * S[:tgt_dim]) @ Vh[:tgt_dim, :]
        return W_compressed.reshape(*W.shape[:-1], tgt_dim).transpose(axis, -1)

    def _pad_expand(self, W: Tensor, tgt_dim: int, axis: int) -> Tensor:
        pad_size = tgt_dim - W.shape[axis]
        # Initialize expansion with small random values (1% of source std)
        padding = torch.randn(*W.shape[:axis], pad_size, *W.shape[axis+1:])
        padding *= W.std() * 0.01
        return torch.cat([W, padding], dim=axis)
```

---

## 4. Calibration Specifications

### 4.1 Zero-Shot Calibration (No Data Required)

```python
class ZeroShotCalibrator:
    """
    Improves converted weights without any training data.
    Uses statistical properties of activations to correct biases.
    """

    def calibrate(
        self,
        source_model,
        converted_model,
        n_samples: int = 1000,
        method: str = "activation_matching"  # activation_matching | output_matching
    ) -> None:
        
        if method == "activation_matching":
            # Generate synthetic inputs from the model's own embedding space
            synthetic_inputs = self._generate_synthetic_inputs(n_samples)
            
            # Collect intermediate activations from both models
            src_acts = self._collect_activations(source_model, synthetic_inputs)
            tgt_acts = self._collect_activations(converted_model, synthetic_inputs)
            
            # Compute per-layer correction factors (no gradients needed)
            for layer_name in src_acts:
                src_mean, src_std = src_acts[layer_name].mean(0), src_acts[layer_name].std(0)
                tgt_mean, tgt_std = tgt_acts[layer_name].mean(0), tgt_acts[layer_name].std(0)
                
                # Apply affine correction: scale and shift target activations to match source
                scale = src_std / (tgt_std + 1e-8)
                bias = src_mean - scale * tgt_mean
                
                self._apply_correction(converted_model, layer_name, scale, bias)
```

### 4.2 Few-Shot Calibration

```python
class FewShotCalibrator:
    """
    Fine-tunes converted weights on a small dataset using gradient-free optimization.
    No backpropagation — uses CMA-ES (evolutionary strategy).
    """

    def calibrate(
        self,
        converted_model,
        calibration_data: DataLoader,
        n_iterations: int = 100,
        budget_tokens: int = 1_000_000   # Total token budget
    ) -> None:
        
        # Only optimize the most sensitive layers (norms + output projections)
        tunable_params = self._get_sensitive_params(converted_model)
        
        # CMA-ES optimization (gradient-free, good for small parameter counts)
        optimizer = CMAES(
            initial_params=tunable_params,
            sigma0=0.01,
            population_size=16
        )
        
        for iteration in range(n_iterations):
            candidates = optimizer.ask()
            fitnesses = []
            
            for candidate in candidates:
                self._apply_params(converted_model, candidate)
                loss = self._evaluate(converted_model, calibration_data)
                fitnesses.append(-loss)  # CMA-ES maximizes
            
            optimizer.tell(candidates, fitnesses)
```

---

## 5. Validation Specifications

### 5.1 Quality Metrics

```python
@dataclass
class ValidationReport:
    # Language modeling quality
    source_perplexity: float        # Baseline
    converted_perplexity: float     # After conversion
    perplexity_ratio: float         # converted/source (closer to 1.0 is better)
    
    # Output similarity
    output_cosine_similarity: float  # Range [0, 1]; 1.0 = identical outputs
    output_token_overlap: float      # Fraction of same top tokens
    
    # Representational similarity
    rsa_score: float                # Representational Similarity Analysis
    layer_correlations: List[float] # Per-layer correlation with source
    
    # Task performance (if benchmark specified)
    task_scores: Dict[str, float]   # {task_name: score}
    
    # Quality tier
    quality_tier: str               # "excellent" | "good" | "acceptable" | "degraded"
    recommendation: str             # Human-readable recommendation
    
    # Conversion metadata
    conversion_time_seconds: float
    source_params: int
    target_params: int
    layers_converted: int
    layers_initialized: int         # Layers with no source equivalent (Xavier init)
```

### 5.2 Quality Tier Thresholds

```python
QUALITY_TIERS = {
    "excellent": {
        "perplexity_ratio": (0.0, 1.15),    # <15% perplexity increase
        "output_similarity": (0.85, 1.0),
        "rsa_score": (0.80, 1.0),
    },
    "good": {
        "perplexity_ratio": (1.15, 1.30),   # 15–30% increase
        "output_similarity": (0.70, 0.85),
        "rsa_score": (0.65, 0.80),
    },
    "acceptable": {
        "perplexity_ratio": (1.30, 1.60),   # 30–60% increase
        "output_similarity": (0.55, 0.70),
        "rsa_score": (0.50, 0.65),
    },
    "degraded": {
        "perplexity_ratio": (1.60, float('inf')),
        "output_similarity": (0.0, 0.55),
        "rsa_score": (0.0, 0.50),
    }
}
```

---

## 6. Performance Specifications

### 6.1 Target Conversion Times

| Model Size | Same-Arch Conversion | Cross-Arch Conversion | With Calibration |
|---|---|---|---|
| 1B params | ~2 min | ~5 min | ~15 min |
| 7B params | ~8 min | ~20 min | ~1 hour |
| 13B params | ~15 min | ~40 min | ~2 hours |
| 70B params | ~60 min | ~2.5 hours | ~8 hours |

*On 32-core CPU. GPU calibration is 5–10× faster.*

### 6.2 Memory Usage

```
Peak RAM during conversion ≈ 2.5 × source_model_size_on_disk

Example:
  Mistral 7B (fp16) on disk: ~14GB
  Peak Paradom RAM usage: ~35GB

Streaming mode peak: ≈ 1.2 × source_model_size (layer-by-layer)
```

---

## 7. Error Handling Specifications

```python
class ParadomException(Exception): pass

class UnsupportedMappingError(ParadomException):
    """Raised when no mapping exists for the source→target pair."""

class DimensionMismatchError(ParadomException):
    """Raised when dimensions cannot be resolved with any strategy."""

class QualityBelowThresholdError(ParadomException):
    """Raised when conversion quality falls below minimum acceptable tier."""
    def __init__(self, tier: str, metrics: ValidationReport):
        self.tier = tier
        self.metrics = metrics

class InsufficientMemoryError(ParadomException):
    """Raised when system RAM is insufficient; suggests streaming mode."""
```

---

## 8. Configuration File Specification

```yaml
# paradom_config.yaml — Global Paradom configuration

conversion:
  default_dtype: "bfloat16"
  streaming_threshold_gb: 32      # Switch to streaming if model > 32GB
  max_workers: 8                  # Parallel layer processing
  checkpoint_interval: 10         # Save intermediate results every N layers

decomposition:
  svd_energy_threshold: 0.99      # Keep 99% of signal energy in SVDs
  attention_rank_fraction: 0.75   # Use 75% of full rank for attention mapping
  
calibration:
  default_method: "activation_matching"
  zero_shot_samples: 1000
  few_shot_max_tokens: 1_000_000
  
validation:
  run_on_complete: true
  test_corpus: "wikitext2"
  test_tokens: 256000
  fail_below_tier: "degraded"     # Raise error if quality is worse than this

output:
  format: "safetensors"
  save_conversion_report: true
  save_intermediate: false        # Set true for debugging

logging:
  level: "INFO"
  progress_bars: true
```
