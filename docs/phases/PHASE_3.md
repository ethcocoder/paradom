# Phase 3: Multi-Paradigm Expansion

**Document:** PARADOM-PHASE-003
**Duration:** Months 7–9
**Prerequisites:** Phase 2 complete. LLM swap engine validated.
**Goal:** Expand Paradom beyond LLMs into Vision, RL, Generative, and Graph paradigms — proving the universal weight equivalence principle across all major ML/DL domains.

---

## Phase 3 Objective

> The LLM paradigm was the proof. Phase 3 is the universalization. Every major ML/DL learning scenario must be handled by the same core swap engine, with paradigm-specific mapping handlers layered on top.

---

## Success Criteria

- [ ] Vision paradigm: CNN → ViT swap working, ≥ 65% Top-1 accuracy retention
- [ ] RL paradigm: DQN → PPO state encoder swap working, ≥ 60% reward retention
- [ ] Generative paradigm: Diffusion → Flow Matching swap working, FID within 30% of source
- [ ] Graph paradigm: GNN → GraphTransformer swap working, ≥ 60% node classification retention
- [ ] Multimodal: CLIP vision encoder → custom arch swap working
- [ ] All paradigms handled by the SAME core swap engine (no paradigm-specific pipelines)
- [ ] `paradom list-paradigms` shows all 5 paradigms
- [ ] Cross-paradigm benchmark report published

---

## Month 7: Vision Paradigm — CNN ↔ ViT

### The Core Challenge

CNNs and ViTs process images in fundamentally different ways:

```
CNN:  slides a small filter over the image → local spatial features
ViT:  splits image into patches, runs attention → global relationships

The weights encode different mathematical operations.
But the learned CONCEPTS (edges, textures, objects) are the same.
That is where Paradom finds the equivalence.
```

### CNN → ViT Mapping Implementation

```python
class CNNToViTMapper(BaseMapper):
    """
    Maps CNN convolutional layers to ViT attention/MLP layers.

    Key insight from research:
      CNN layer depth correlates with abstraction level.
      ViT layer depth also correlates with abstraction level.
      Matching by depth (early/mid/late) gives functional alignment.

    Depth alignment:
      CNN early layers (conv1-3)  → ViT patch embedding + early attention
      CNN mid layers   (conv4-6)  → ViT mid attention layers
      CNN late layers  (conv7+)   → ViT late MLP layers
      CNN classifier             → ViT classification head
    """

    def map_conv_to_attention(
        self,
        W_conv: Tensor,    # Shape: (C_out, C_in, H, W)
        target_dim: int    # ViT hidden dimension
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Maps a CNN convolutional filter to ViT Q, K, V weight matrices.

        Strategy:
          1. Tucker-decompose the conv filter → core tensor G
          2. G encodes the essential spatial pattern (what to look for)
          3. Reshape G into three projections (Q, K, V)
             Q encodes "what am I looking for?"
             K encodes "what do I offer?"
             V encodes "what do I pass along?"
          4. Project each to target_dim
        """
        from paradom.math.tucker import tucker_decompose

        # Step 1: Decompose conv filter
        # Remove spatial dims by flattening H×W into feature space
        W_flat = W_conv.reshape(W_conv.shape[0], -1)  # (C_out, C_in×H×W)

        # Step 2: SVD to extract principal directions
        U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
        rank = min(target_dim, len(S))

        # Step 3: Build Q, K, V from different subspaces of the singular structure
        # Q: uses left singular vectors (output feature directions)
        W_Q = U[:, :rank].T                          # (rank, C_out) → project to (target_dim, target_dim)
        W_Q = self._project_to_dim(W_Q, target_dim)

        # K: uses right singular vectors (input feature directions)
        W_K = Vh[:rank, :]                           # (rank, C_in×H×W)
        W_K = self._project_to_dim(W_K, target_dim)

        # V: uses singular value-weighted combination (the "important signal")
        W_V = (U[:, :rank] * S[:rank]).T
        W_V = self._project_to_dim(W_V, target_dim)

        return W_Q, W_K, W_V

    def map_batchnorm_to_layernorm(
        self,
        bn_weight: Tensor,   # BatchNorm scale (gamma)
        bn_bias: Tensor,     # BatchNorm shift (beta)
    ) -> Tuple[Tensor, Tensor]:
        """
        BatchNorm and LayerNorm both normalize and rescale activations.
        Their weight (gamma) and bias (beta) parameters are directly equivalent.
        This is always a direct swap — no projection needed.
        """
        # Direct swap: same functional role, same parameter meaning
        # Drop running_mean, running_var (not parameters, not swapped)
        return bn_weight.clone(), bn_bias.clone()
```

### Vision Benchmark Protocol

```bash
# Evaluate CNN → ViT swap on ImageNet validation set
paradom validate \
  --source torchvision/resnet50 \
  --swapped ./output/resnet50_as_vit \
  --paradigm vision \
  --benchmark top1_accuracy,top5_accuracy,cka_layers \
  --dataset imagenet_val \
  --report ./benchmarks/phase3_cnn_to_vit.json

# Expected output:
# ResNet-50 baseline Top-1:        76.1%
# ViT-S/16 from scratch (trained): 79.8%
# ViT-S/16 from Paradom swap:      51-62%  ← target: ≥65%
# CKA layer similarity:            0.52 avg
```

---

## Month 8: RL Paradigm — DQN ↔ PPO

### Why RL Weight Transfer Matters

Training an RL agent from scratch is expensive:
- Millions of environment interactions
- Hours to days of wall-clock time
- High sensitivity to hyperparameters

If Paradom can transfer the learned **state representation** from a DQN to a PPO policy, the PPO agent starts from a much better position — saving 60-80% of early training time.

### DQN → PPO Mapping Implementation

```python
class DQNToPPOMapper(BaseMapper):
    """
    Transfers state-encoding weights from a trained DQN agent
    to a PPO policy network.

    What transfers:
      The DQN feature extractor has learned to represent states efficiently.
      This representation is architecture-agnostic — it describes the environment.
      PPO can start with this representation instead of learning from scratch.

    What does NOT transfer:
      DQN Q-value head → PPO action logits (different output semantics)
      DQN has no value head → PPO value head (initialize from scratch)
    """

    def map_feature_extractor(
        self,
        dqn_features: Dict[str, Tensor],  # DQN feature extractor weights
        ppo_feature_config: dict           # PPO feature extractor architecture
    ) -> Dict[str, Tensor]:
        """
        The feature extractor is the highest-value swap in RL transfer.
        It encodes: "how to understand this environment's states."
        """
        mapped = {}

        for layer_name, W in dqn_features.items():
            target_name = self._map_layer_name(layer_name)
            target_shape = ppo_feature_config[target_name]["shape"]

            if W.shape == target_shape:
                # Direct swap — same architecture feature extractor
                mapped[target_name] = W.clone()
            else:
                # Projected swap — different width but same depth
                mapped[target_name] = self.projected_swap(W, target_shape)

        return mapped

    def map_q_to_policy(
        self,
        W_q_head: Tensor,    # DQN Q-value head: (n_actions, hidden_dim)
        n_actions: int,
        hidden_dim: int
    ) -> Tensor:
        """
        Q-values and policy logits both score actions, but differently:
          Q(s, a) = expected cumulative reward (unbounded real number)
          π(a|s)  = log probability of action (negative, softmax normalized)

        Partial transfer: the direction of action preference is preserved,
        but the scale and offset must be corrected.

        Correction: normalize Q-values → approximate log-probabilities
        """
        # Center and scale Q-values to approximate log-probability range
        W_normalized = (W_q_head - W_q_head.mean(dim=0)) / (W_q_head.std(dim=0) + 1e-8)
        # Apply temperature scaling (Q-values are typically larger magnitude)
        W_policy_init = W_normalized * 0.1
        return W_policy_init

    def initialize_value_head(self, hidden_dim: int) -> Tensor:
        """
        DQN has no value head — PPO requires one.
        Initialize fresh with Xavier uniform (standard practice).
        """
        W = torch.empty(1, hidden_dim)
        nn.init.xavier_uniform_(W)
        return W
```

### RL Validation Protocol

```python
def validate_rl_transfer(
    source_dqn_path: str,
    converted_ppo_path: str,
    env_name: str,            # e.g. "CartPole-v1", "Atari/Breakout"
    n_eval_episodes: int = 100
) -> RLValidationReport:
    """
    Key metrics:
      1. Initial episode reward (without any further training)
         Measures: how much intelligence transferred directly

      2. Steps to baseline reward (with continued PPO training)
         Measures: how much training time was saved vs training from scratch

      3. Policy entropy (how confident is the transferred policy)
         Measures: whether the policy is degenerate or well-calibrated
    """
    ...
```

---

## Month 9: Generative + Graph + Multimodal Paradigms

### Diffusion → Flow Matching

```python
class DiffusionToFlowMapper(BaseMapper):
    """
    Stable Diffusion → Flow Matching architecture swap.

    Mathematical basis:
      Both model the same data distribution p(x).
      Diffusion: learns score function ∇log p(x_t)
      Flow:      learns velocity field v(x_t, t) pointing toward data

      Under DDPM parameterization: v ≈ -ε  (flow ≈ negative noise)
      Therefore: W_flow_output = -W_diffusion_output (sign flip on output layer)
      All other weights transfer directly or with projection.
    """

    DIRECT_SWAP_LAYERS = [
        "encoder.*",          # Image encoder (identical role)
        "decoder.*",          # Image decoder (identical role)
        "time_embed.*",       # Time conditioning (identical role)
        "cross_attn.*",       # Text cross-attention (identical role)
    ]

    def map_output_layer(self, W_noise_pred: Tensor) -> Tensor:
        """
        Noise predictor → flow predictor:
        Flow velocity = approximately -noise direction
        """
        return -W_noise_pred.clone()  # Sign flip is the key operation
```

### GNN → GraphTransformer

```python
class GNNToGraphTransformerMapper(BaseMapper):
    """
    Graph Neural Network → Graph Transformer.

    Conceptual equivalence:
      GNN message passing: aggregate neighbor features weighted by edge structure
      Graph attention:     aggregate neighbor features weighted by learned attention

      Both learn: "how to combine information from neighboring nodes"
      The weights encoding this combination are translatable.

    Layer correspondence:
      GNN node transform   → GraphTransformer Q/K/V projections
      GNN edge transform   → GraphTransformer edge bias terms
      GNN aggregation      → GraphTransformer attention output
      GNN readout          → GraphTransformer pooling head
    """

    def map_message_passing_to_attention(
        self,
        W_message: Tensor,   # GNN message function weights
        W_update: Tensor,    # GNN update function weights
        target_dim: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        GNN message passing has no direct Q/K/V decomposition.
        We split the message weight matrix into 3 equal parts
        to initialize Q, K, V projections.
        """
        # SVD to find principal directions
        U, S, Vh = torch.linalg.svd(W_message, full_matrices=False)
        third = len(S) // 3

        # Divide singular value spectrum into Q, K, V subspaces
        W_Q = self._reconstruct(U, S[:third],   Vh[:third, :], target_dim)
        W_K = self._reconstruct(U, S[third:2*third], Vh[third:2*third, :], target_dim)
        W_V = self._reconstruct(U, S[2*third:], Vh[2*third:, :], target_dim)
        W_O = self._project_to_dim(W_update, target_dim)

        return W_Q, W_K, W_V, W_O
```

### Phase 3 Exit Criteria

1. ✅ All 5 paradigms (LLM, Vision, RL, Generative, Graph) handled
2. ✅ Quality targets met for each paradigm
3. ✅ Single unified CLI handles all paradigms (`--paradigm` flag)
4. ✅ Cross-paradigm benchmark report published
5. ✅ Phase 4 production plan finalized
