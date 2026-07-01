# System Architecture: Paradom Framework

**Document:** PARADOM-ARCH-001  
**Version:** 2.0.0  
**Date:** 2026-06-30

---

## 1. Architectural Philosophy

Paradom is built around three non-negotiable design principles:

**Principle 1 — Swap, Don't Recalculate**
Never reconstruct what can be directly translated. Move the numbers, not the math.

**Principle 2 — Surgical Precision**
Only touch the weights that carry intelligence. Leave the rest to sensible defaults.

**Principle 3 — Universal Paradigm Support**
The same core engine must handle LLMs, CNNs, RL policies, diffusion models, and GNNs without paradigm-specific rewrites.

---

## 2. High-Level System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PARADOM FRAMEWORK                                   ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │                         INPUT LAYER                                    │  ║
║  │                                                                        │  ║
║  │  HuggingFace Hub  │  Local Files  │  Kaggle  │  Custom Sources         │  ║
║  │  (LLaMA, Mistral, Falcon, Gemma, DQN agents, Diffusion models, GNNs)  │  ║
║  └───────────────────────────────┬────────────────────────────────────────┘  ║
║                                  │                                           ║
║                                  ▼                                           ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │                      PARADOM CORE ENGINE                               │  ║
║  │                                                                        │  ║
║  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                 │  ║
║  │   │   LOADER    │──▶│   PARSER    │──▶│  IMPORTANCE │                 │  ║
║  │   │             │   │             │   │   SCORER    │                 │  ║
║  │   │ Multi-format│   │ Paradigm    │   │             │                 │  ║
║  │   │ weight load │   │ detection & │   │ Find the    │                 │  ║
║  │   │             │   │ layer group │   │ "winning    │                 │  ║
║  │   │             │   │ extraction  │   │  tickets"   │                 │  ║
║  │   └─────────────┘   └─────────────┘   └──────┬──────┘                 │  ║
║  │                                               │                        │  ║
║  │                                               ▼                        │  ║
║  │   ┌────────────────────────────────────────────────────────────────┐   │  ║
║  │   │                  EQUIVALENCE IDENTIFIER                        │   │  ║
║  │   │                                                                │   │  ║
║  │   │  "Which weight in Model A is the equivalent product            │   │  ║
║  │   │   of which weight in Model B?"                                 │   │  ║
║  │   │                                                                │   │  ║
║  │   │  Methods: CKA similarity │ SVD alignment │ Functional matching │   │  ║
║  │   └───────────────────────────────────┬────────────────────────────┘   │  ║
║  │                                       │                                │  ║
║  │                                       ▼                                │  ║
║  │   ┌────────────────────────────────────────────────────────────────┐   │  ║
║  │   │                     SWAP ENGINE                                │   │  ║
║  │   │                                                                │   │  ║
║  │   │  Direct Swap  │  Projected Swap  │  Tensor Decomp Swap        │   │  ║
║  │   │  (same dims)  │  (diff dims)     │  (CNN ↔ ViT etc.)          │   │  ║
║  │   └───────────────────────────────────┬────────────────────────────┘   │  ║
║  │                                       │                                │  ║
║  │                                       ▼                                │  ║
║  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                 │  ║
║  │   │ CONSTRUCTOR │──▶│  VALIDATOR  │──▶│   REPORTER  │                 │  ║
║  │   │             │   │             │   │             │                 │  ║
║  │   │ Assemble    │   │ Quality     │   │ Swap report │                 │  ║
║  │   │ target      │   │ measurement │   │ + metrics   │                 │  ║
║  │   │ checkpoint  │   │ + RSA score │   │             │                 │  ║
║  │   └─────────────┘   └─────────────┘   └─────────────┘                 │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                  │                                           ║
║                                  ▼                                           ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │                         OUTPUT LAYER                                   │  ║
║  │                                                                        │  ║
║  │  Custom Arch  │  Mamba SSM  │  MoE  │  ViT  │  PPO Policy  │  Flow    │  ║
║  │                                                                        │  ║
║  │  SOVEREIGN AI DEPLOYMENT — runs on consumer hardware, no cloud needed  │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Module Breakdown

### 3.1 Package Structure

```
paradom/
│
├── core/
│   ├── loader.py           # Multi-format, multi-paradigm weight loader
│   ├── parser.py           # Paradigm detector & layer group extractor
│   ├── importance.py       # Importance scoring (find the winning tickets)
│   ├── equivalence.py      # Equivalence identifier (CKA, SVD alignment)
│   ├── swap_engine.py      # The actual swap operations
│   ├── constructor.py      # Target model assembly
│   └── validator.py        # Quality measurement
│
├── paradigms/
│   ├── base.py             # Abstract paradigm definition
│   ├── llm.py              # LLM paradigm (Transformer, Mamba, MoE)
│   ├── vision.py           # Vision paradigm (CNN, ViT, MLP-Mixer)
│   ├── rl.py               # RL paradigm (DQN, PPO, Actor-Critic)
│   ├── generative.py       # Generative paradigm (Diffusion, Flow Matching)
│   ├── graph.py            # Graph paradigm (GNN, GraphTransformer)
│   └── multimodal.py       # Multimodal paradigm (CLIP, LLaVA)
│
├── mappings/
│   ├── registry.py         # Central mapping strategy registry
│   ├── llm/
│   │   ├── transformer_to_mamba.py
│   │   ├── transformer_to_moe.py
│   │   └── transformer_to_transformer.py
│   ├── vision/
│   │   ├── cnn_to_vit.py
│   │   └── vit_to_cnn.py
│   ├── rl/
│   │   ├── dqn_to_ppo.py
│   │   └── actor_critic_to_transformer.py
│   └── generative/
│       └── diffusion_to_flow.py
│
├── math/
│   ├── svd.py              # SVD operations (blocked, randomized)
│   ├── cka.py              # Centered Kernel Alignment
│   ├── procrustes.py       # Weight space alignment
│   ├── tucker.py           # Tucker decomposition for CNN tensors
│   └── ot.py               # Optimal transport for distribution matching
│
├── cli/
│   └── paradom_cli.py      # Command-line interface
│
├── api/
│   └── paradom_api.py      # Python API
│
└── benchmarks/
    ├── llm_bench.py         # Perplexity, MMLU, HellaSwag
    ├── vision_bench.py      # Top-1/5 accuracy, transfer benchmarks
    ├── rl_bench.py          # Episode reward, policy similarity
    └── similarity.py        # RSA, CKA across all paradigms
```

---

## 4. The Equivalence Identifier — Core Module

This is the most important module in Paradom. Everything else depends on correctly answering: **"Which weight in Model A corresponds to which weight in Model B?"**

### 4.1 Equivalence Detection Methods

**Method 1: Functional Role Matching (Primary)**
```python
class FunctionalRoleMatcher:
    """
    Matches weights by their functional role in the network,
    not by their position or name.
    
    Example:
      LLaMA's "model.layers.0.self_attn.q_proj.weight"
      and Mamba's "backbone.layers.0.mixer.in_proj.weight"
      
      Both project input into a space used for context selection.
      They are functionally equivalent despite different names.
    """
    
    FUNCTIONAL_ROLES = {
        "context_selection_query": [...],  # Q matrices, SSM B matrices
        "context_selection_key":   [...],  # K matrices, SSM C matrices  
        "value_aggregation":       [...],  # V matrices, SSM output proj
        "feed_forward_expand":     [...],  # FFN up/gate projections
        "feed_forward_contract":   [...],  # FFN down projections
        "normalization":           [...],  # LayerNorm, RMSNorm weights
        "embedding":               [...],  # Token, patch, node embeddings
        "output_head":             [...],  # Final projection to vocab/classes
    }
```

**Method 2: CKA Similarity (Validation)**
```python
class CKASimilarityMatcher:
    """
    Uses Centered Kernel Alignment to measure representational similarity.
    Confirms that functionally matched weights produce similar activations.
    
    CKA score > 0.7: Strong equivalence (safe to swap)
    CKA score 0.4-0.7: Moderate equivalence (swap with projection)
    CKA score < 0.4: Weak equivalence (swap with caution, use OT)
    """
    
    def compute_cka(
        self,
        activations_A: Tensor,   # (n_samples × d_A)
        activations_B: Tensor,   # (n_samples × d_B)
    ) -> float:
        # Linear CKA: fast, reliable for same-dimension comparison
        return self._linear_cka(activations_A, activations_B)
```

**Method 3: SVD Alignment (For Different Dimensions)**
```python
class SVDAlignmentMatcher:
    """
    When source and target have different dimensions,
    align them by finding the optimal rotation/projection
    between their singular vector spaces.
    
    Based on: Procrustes analysis in weight space
    """
    
    def align(self, W_source: Tensor, W_target_shape: tuple) -> Tensor:
        U_s, S_s, Vh_s = torch.linalg.svd(W_source, full_matrices=False)
        # Project source singular vectors to target dimension
        # Preserve the directions of maximum variance (most important numbers)
        ...
```

---

## 5. The Swap Engine

### 5.1 Swap Types

**Type 1: Direct Swap**
```python
# When source and target weight have identical shape
W_target = W_source.clone()
```
Used for: same-family LLM conversion, embedding tables with same vocab size

**Type 2: Projected Swap**
```python
# When dimensions differ but functional role is equivalent
# Find optimal linear projection from source to target space
P = compute_procrustes_projection(W_source, W_target_shape)
W_target = P @ W_source @ P.T
```
Used for: LLM hidden dimension changes, cross-architecture with different widths

**Type 3: Tensor Decomposition Swap**
```python
# For multi-dimensional tensors (CNN filters)
# Decompose source, recompose in target structure
G, factors = tucker_decompose(W_source_4d)
W_target = tucker_recompose(G, target_factors)
```
Used for: CNN ↔ ViT, any 3D/4D weight tensor translation

**Type 4: Distribution-Matched Swap**
```python
# When direct equivalence is weak — use Optimal Transport
# Find the transport map that moves source weight distribution
# to target weight distribution most faithfully
T = compute_wasserstein_transport(W_source_dist, W_target_dist)
W_target = apply_transport(T, W_source)
```
Used for: GNN ↔ Transformer, RL paradigm swaps

### 5.2 Importance Scoring — The Winning Ticket Finder

```python
class ImportanceScorer:
    """
    Identifies which weights carry the essential intelligence.
    These are the only weights Paradom needs to swap.
    The rest are initialized from target architecture defaults.
    """

    def score(self, model, method: str = "gradient_sensitivity") -> Dict[str, Tensor]:
        if method == "gradient_sensitivity":
            # Weights with high gradient magnitude on representative inputs
            # are most important for the model's behavior
            return self._gradient_sensitivity(model)
        
        elif method == "activation_magnitude":
            # Weights connected to high-magnitude activations are most active
            return self._activation_magnitude(model)
        
        elif method == "svd_spectrum":
            # Top singular values of each weight matrix carry most variance
            # = most information = most important numbers
            return self._svd_spectrum(model)
        
        elif method == "lottery_ticket":
            # Full lottery ticket identification (magnitude-based pruning)
            return self._lottery_ticket_mask(model, sparsity=0.8)
    
    def get_top_k_weights(self, scores: Dict, k_percent: float = 0.20) -> Dict[str, Tensor]:
        """Return the top k% most important weights — the winning ticket."""
        ...
```

---

## 6. Paradigm Handlers

Each ML/DL paradigm has a specialized handler that knows how to:
1. Parse its specific layer types
2. Assign functional roles to its weights
3. Specify valid swap targets

### 6.1 LLM Paradigm Handler

```python
class LLMParadigmHandler(BaseParadigmHandler):
    SUPPORTED_ARCHITECTURES = ["llama", "mistral", "falcon", "gemma", "mamba", "moe"]
    
    LAYER_ROLES = {
        "q_proj": "context_selection_query",
        "k_proj": "context_selection_key",
        "v_proj": "value_aggregation",
        "o_proj": "value_aggregation",
        "gate_proj": "feed_forward_expand",
        "up_proj": "feed_forward_expand",
        "down_proj": "feed_forward_contract",
        # Mamba equivalents
        "in_proj": "context_selection_query",
        "x_proj": "context_selection_key",
        "out_proj": "value_aggregation",
    }
```

### 6.2 Vision Paradigm Handler

```python
class VisionParadigmHandler(BaseParadigmHandler):
    SUPPORTED_ARCHITECTURES = ["resnet", "vgg", "vit", "deit", "convnext", "mlp_mixer"]
    
    LAYER_ROLES = {
        # CNN layers
        "conv": "spatial_feature_extraction",
        "bn": "normalization",
        # ViT layers
        "patch_embed": "embedding",
        "attn.qkv": "context_selection_query",  # ViT combines Q,K,V
        "attn.proj": "value_aggregation",
        "mlp.fc1": "feed_forward_expand",
        "mlp.fc2": "feed_forward_contract",
    }
```

### 6.3 RL Paradigm Handler

```python
class RLParadigmHandler(BaseParadigmHandler):
    SUPPORTED_ARCHITECTURES = ["dqn", "ppo", "a2c", "sac", "td3", "dreamer"]
    
    LAYER_ROLES = {
        # DQN
        "feature_extractor": "state_encoding",
        "q_net": "action_value_head",
        # PPO
        "policy_net": "state_encoding",
        "action_head": "action_distribution_head",
        "value_head": "value_estimation_head",
    }
    
    # State encoding layers are the most transferable
    # Action heads require architectural translation
    DIRECTLY_SWAPPABLE = ["state_encoding"]
    NEEDS_TRANSLATION = ["action_value_head", "action_distribution_head"]
```

---

## 7. Interface Design

### 7.1 CLI Interface

```bash
# LLM conversion
paradom swap \
  --source meta-llama/Llama-3-8B \
  --target-arch mamba \
  --target-config configs/mamba_custom.yaml \
  --output ./output/llama_as_mamba \
  --importance-method svd_spectrum \
  --swap-fraction 0.20

# Vision model conversion
paradom swap \
  --source torchvision/resnet50 \
  --paradigm vision \
  --target-arch vit \
  --target-config configs/vit_small.yaml \
  --output ./output/resnet_as_vit

# RL policy transfer
paradom swap \
  --source ./trained_dqn_agent.pt \
  --paradigm rl \
  --target-arch ppo \
  --target-config configs/ppo_policy.yaml \
  --output ./output/dqn_to_ppo

# Inspect equivalences (before swapping)
paradom identify \
  --source meta-llama/Llama-3-8B \
  --target-arch mamba \
  --target-config configs/mamba_custom.yaml \
  --report ./reports/equivalence_map.json

# List all supported paradigms and mappings
paradom list-paradigms
paradom list-mappings --paradigm llm
```

### 7.2 Python API

```python
from paradom import Paradom, TargetSpec, SwapConfig

# Configure
target = TargetSpec.from_yaml("configs/mamba_custom.yaml")
config = SwapConfig(
    importance_method="svd_spectrum",
    swap_fraction=0.20,          # Only swap the top 20% most important weights
    swap_type="auto",            # Paradom chooses direct/projected/tensor/OT
    validate_after=True,
    output_format="safetensors"
)

# Run
engine = Paradom()
result = engine.swap(
    source="meta-llama/Llama-3-8B",
    target=target,
    config=config,
    output_path="./output/my_model"
)

# Review
print(f"Weights swapped: {result.weights_swapped} ({result.swap_fraction:.1%})")
print(f"Quality score: {result.quality_score:.3f}")
print(f"Quality tier: {result.quality_tier}")
print(f"Paradigm: {result.source_paradigm} → {result.target_paradigm}")
print(f"Conversion time: {result.time_seconds:.1f}s")
print(f"Peak RAM used: {result.peak_ram_gb:.1f}GB")
```

---

## 8. Memory Architecture (Why It's Lightweight)

```
TRADITIONAL APPROACH (Heavy):
  Load all of Model A (14GB) into RAM
  Load all of Model B (14GB) into RAM  
  Run full computation on both
  Peak RAM: ~40GB+

PARADOM APPROACH (Lightweight):
  Stream Model A layer by layer (~450MB per layer for 7B)
  Score importance → identify top 20% weights (~90MB)
  Load equivalent layer from Model B
  Swap the top 20%
  Write output layer
  Free memory
  Move to next layer
  
  Peak RAM: ~1-2GB
  A standard developer laptop handles a 7B model.
```

---

## 9. Deployment Architecture

```
MINIMUM HARDWARE (Paradom Swap Mode):
  RAM: 4GB (streaming, layer by layer)
  CPU: Any modern quad-core
  GPU: Not required
  Storage: 2× model size
  OS: Linux / macOS / Windows

RECOMMENDED HARDWARE:
  RAM: 16GB (can hold more context for better equivalence detection)
  CPU: 8+ cores (parallel layer processing)
  Storage: SSD, 3× model size

FOR 70B MODELS:
  RAM: 16GB (streaming mode keeps this constant regardless of model size)
  CPU: 16+ cores (parallel processing across layers)
  Time: 2-4 hours (vs days for retraining)
```