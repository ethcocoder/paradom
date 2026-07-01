# Paradom Core Code Scaffold
# This file gives implementers the exact class/function signatures
# to build from. All docstrings specify the contract.

"""
paradom/core/loader.py
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import Tensor


# ─────────────────────────────────────────────
# ENUMS & DATA STRUCTURES
# ─────────────────────────────────────────────

class FunctionalRole(Enum):
    """Universal taxonomy of weight functional roles across all paradigms."""
    # Universal
    EMBEDDING         = "embedding"
    OUTPUT_HEAD       = "output_head"
    NORMALIZATION     = "normalization"
    BIAS              = "bias"
    # Context / Attention
    CONTEXT_QUERY     = "context_query"
    CONTEXT_KEY       = "context_key"
    CONTEXT_VALUE     = "context_value"
    CONTEXT_OUTPUT    = "context_output"
    # Feed-forward
    FFN_EXPAND        = "ffn_expand"
    FFN_CONTRACT      = "ffn_contract"
    # Vision
    SPATIAL_FILTER    = "spatial_filter"
    PATCH_EMBED       = "patch_embed"
    # RL
    STATE_ENCODER     = "state_encoder"
    ACTION_HEAD       = "action_head"
    VALUE_HEAD        = "value_head"
    # Generative
    NOISE_PREDICTOR   = "noise_predictor"
    FLOW_PREDICTOR    = "flow_predictor"
    TIME_EMBED        = "time_embed"
    # Graph
    NODE_TRANSFORM    = "node_transform"
    EDGE_TRANSFORM    = "edge_transform"
    AGGREGATION       = "aggregation"
    # Unknown (must be resolved before swapping)
    UNKNOWN           = "unknown"


class SwapType(Enum):
    DIRECT      = "direct"       # Same shape, high CKA — direct copy
    PROJECTED   = "projected"    # Different shape — SVD projection
    TENSOR      = "tensor"       # Multi-dim (CNN) — Tucker decomp
    OT          = "ot"           # Weak equivalence — optimal transport
    DERIVED     = "derived"      # No source equivalent — mathematically derived
    SKIP        = "skip"         # No equivalent — Xavier init


class QualityTier(Enum):
    EXCELLENT  = "excellent"     # ≥85% retention
    GOOD       = "good"          # 70–85% retention
    ACCEPTABLE = "acceptable"    # 55–70% retention
    DEGRADED   = "degraded"      # <55% retention


@dataclass
class WeightProduct:
    """A single weight tensor with full metadata."""
    name: str
    tensor: Tensor
    shape: tuple
    functional_role: FunctionalRole
    paradigm: str
    architecture: str
    layer_index: int             # Which transformer/conv layer this belongs to
    importance_score: float = 0.0
    dtype: torch.dtype = torch.float16


@dataclass
class EquivalencePair:
    """A matched pair of source and target weight positions."""
    source: WeightProduct
    target_layer_name: str
    target_shape: tuple
    cka_score: float
    swap_type: SwapType
    confidence: float

    @property
    def is_safe_to_swap(self) -> bool:
        return (
            self.source.functional_role != FunctionalRole.UNKNOWN and
            self.cka_score > 0.20 and
            self.confidence > 0.50
        )


@dataclass
class EquivalenceMap:
    """Complete mapping between source and target model."""
    source_model: str
    target_architecture: str
    pairs: List[EquivalencePair]
    unmapped_source: List[str]        # Source layers with no target equivalent
    uninitialized_target: List[str]   # Target layers with no source (Xavier init)
    mean_cka: float
    estimated_quality_tier: QualityTier


@dataclass
class SwapValidationReport:
    """Full report generated after a swap operation."""
    source_model: str
    target_architecture: str
    source_paradigm: str
    target_paradigm: str
    total_weights: int
    weights_swapped: int
    swap_fraction: float
    swap_type_distribution: Dict[str, float]
    cka_scores: Dict[str, float]
    mean_cka: float
    paradigm_metric_name: str
    source_paradigm_metric: float
    converted_paradigm_metric: float
    retention_fraction: float
    quality_tier: QualityTier
    recommendation: str
    conversion_time_seconds: float
    peak_ram_mb: float


# ─────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────

class ModelLoader:
    """
    Loads model weights from any supported format into a
    streaming iterator of WeightProduct objects.

    Design: yields one layer at a time — never loads the full model.
    This keeps RAM constant at ~1-2GB regardless of model size.
    """

    SUPPORTED_FORMATS = ["safetensors", "pytorch", "gguf", "huggingface_hub"]

    def stream_layers(
        self,
        source: str,              # HF model ID or local path
        cache_dir: Optional[str] = None
    ):
        """
        Generator that yields WeightProduct objects one layer at a time.

        Usage:
            loader = ModelLoader()
            for weight in loader.stream_layers("meta-llama/Llama-3-8B"):
                process(weight)
                # weight is freed from memory before next iteration
        """
        format_ = self._detect_format(source)

        if format_ == "huggingface_hub":
            yield from self._stream_from_hub(source, cache_dir)
        elif format_ == "safetensors":
            yield from self._stream_safetensors(source)
        elif format_ == "pytorch":
            yield from self._stream_pytorch(source)
        elif format_ == "gguf":
            yield from self._stream_gguf(source)
        else:
            raise ValueError(f"Unsupported format: {format_}")

    def _detect_format(self, source: str) -> str:
        """Auto-detect source format from path or model ID pattern."""
        ...

    def _stream_from_hub(self, model_id: str, cache_dir: Optional[str]):
        """Download and stream from HuggingFace Hub."""
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(model_id, cache_dir=cache_dir)
        yield from self._stream_safetensors(local_path)

    def _stream_safetensors(self, path: str):
        """Stream layers from .safetensors file(s)."""
        from safetensors import safe_open
        ...

    def get_config(self, source: str) -> Dict[str, Any]:
        """Load model config.json without loading weights."""
        ...


# ─────────────────────────────────────────────
# FUNCTIONAL ROLE MATCHER
# ─────────────────────────────────────────────

class FunctionalRoleMatcher:
    """
    Assigns a FunctionalRole to each weight tensor based on its
    layer name and architecture.

    This is the foundation of the equivalence identifier —
    two weights with the same FunctionalRole are candidates for swapping.
    """

    # Mapping from (architecture, layer_name_pattern) → FunctionalRole
    ROLE_PATTERNS: Dict[str, Dict[str, FunctionalRole]] = {
        "llama": {
            "embed_tokens":     FunctionalRole.EMBEDDING,
            "q_proj":           FunctionalRole.CONTEXT_QUERY,
            "k_proj":           FunctionalRole.CONTEXT_KEY,
            "v_proj":           FunctionalRole.CONTEXT_VALUE,
            "o_proj":           FunctionalRole.CONTEXT_OUTPUT,
            "gate_proj":        FunctionalRole.FFN_EXPAND,
            "up_proj":          FunctionalRole.FFN_EXPAND,
            "down_proj":        FunctionalRole.FFN_CONTRACT,
            "input_layernorm":  FunctionalRole.NORMALIZATION,
            "post_attention_layernorm": FunctionalRole.NORMALIZATION,
            "lm_head":          FunctionalRole.OUTPUT_HEAD,
        },
        "mamba": {
            "embedding":        FunctionalRole.EMBEDDING,
            "in_proj":          FunctionalRole.CONTEXT_QUERY,
            "x_proj":           FunctionalRole.CONTEXT_KEY,
            "out_proj":         FunctionalRole.CONTEXT_OUTPUT,
            "dt_proj":          FunctionalRole.FFN_EXPAND,
            "norm":             FunctionalRole.NORMALIZATION,
            "lm_head":          FunctionalRole.OUTPUT_HEAD,
        },
        "resnet": {
            "conv1":            FunctionalRole.SPATIAL_FILTER,
            "layer1":           FunctionalRole.SPATIAL_FILTER,
            "layer2":           FunctionalRole.SPATIAL_FILTER,
            "layer3":           FunctionalRole.SPATIAL_FILTER,
            "layer4":           FunctionalRole.SPATIAL_FILTER,
            "bn":               FunctionalRole.NORMALIZATION,
            "fc":               FunctionalRole.OUTPUT_HEAD,
        },
        "vit": {
            "patch_embed":      FunctionalRole.PATCH_EMBED,
            "attn.qkv":         FunctionalRole.CONTEXT_QUERY,
            "attn.proj":        FunctionalRole.CONTEXT_OUTPUT,
            "mlp.fc1":          FunctionalRole.FFN_EXPAND,
            "mlp.fc2":          FunctionalRole.FFN_CONTRACT,
            "norm":             FunctionalRole.NORMALIZATION,
            "head":             FunctionalRole.OUTPUT_HEAD,
        },
    }

    def assign_role(
        self,
        layer_name: str,
        architecture: str
    ) -> FunctionalRole:
        """Assign FunctionalRole based on layer name and architecture."""
        patterns = self.ROLE_PATTERNS.get(architecture, {})
        for pattern, role in patterns.items():
            if pattern in layer_name:
                return role
        return FunctionalRole.UNKNOWN


# ─────────────────────────────────────────────
# IMPORTANCE SCORER
# ─────────────────────────────────────────────

class ImportanceScorer:
    """
    Identifies which weights carry the essential intelligence.
    Only these weights are swapped — the rest are initialized fresh.
    This is the "winning ticket" finder.
    """

    def score_svd_spectrum(
        self,
        W: Tensor,
        top_k_fraction: float = 0.20
    ) -> Tensor:
        """
        Returns a boolean mask of the top-k% most important weights
        by their contribution to the singular value spectrum.

        No data required. Fast. Mathematically grounded.
        Recommended for most use cases.

        Args:
            W: Weight tensor (2D for linear layers)
            top_k_fraction: Fraction of weights to mark as important

        Returns:
            Boolean mask, same shape as W
        """
        if W.dim() == 1:
            # Bias or 1D weight — all equally important
            return torch.ones_like(W, dtype=torch.bool)

        W_2d = W.reshape(W.shape[0], -1).float()
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

        # Importance of each element = product of left and right singular contributions
        # Elements that participate in top singular directions are most important
        rank = max(1, int(len(S) * top_k_fraction))
        importance = (
            U[:, :rank].abs().sum(dim=1, keepdim=True) *
            Vh[:rank, :].abs().sum(dim=0, keepdim=True)
        )
        importance = importance.reshape(W.shape)

        # Create mask: top k_fraction% by importance score
        threshold = importance.flatten().quantile(1 - top_k_fraction)
        return importance >= threshold

    def score_lottery_ticket(
        self,
        W: Tensor,
        sparsity: float = 0.80
    ) -> Tensor:
        """
        Classic magnitude-based importance: keep top (1-sparsity)% by |W|.
        Direct implementation of the Lottery Ticket Hypothesis.
        """
        threshold = W.abs().flatten().quantile(sparsity)
        return W.abs() >= threshold


# ─────────────────────────────────────────────
# SWAP ENGINE
# ─────────────────────────────────────────────

class SwapEngine:
    """
    Executes the actual weight swap operations.
    Called by the main Paradom pipeline for each equivalence pair.
    """

    def swap(
        self,
        source_weight: Tensor,
        target_shape: tuple,
        swap_type: SwapType,
        importance_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Execute a swap operation.

        If importance_mask is provided, only swaps the masked weights.
        Remaining target weights are Xavier-initialized.
        """
        if swap_type == SwapType.DIRECT:
            return self._direct_swap(source_weight, target_shape, importance_mask)
        elif swap_type == SwapType.PROJECTED:
            return self._projected_swap(source_weight, target_shape, importance_mask)
        elif swap_type == SwapType.TENSOR:
            return self._tensor_swap(source_weight, target_shape)
        elif swap_type == SwapType.OT:
            return self._ot_swap(source_weight, target_shape)
        elif swap_type == SwapType.SKIP:
            return self._xavier_init(target_shape, source_weight.dtype)
        else:
            raise ValueError(f"Unknown swap type: {swap_type}")

    def _direct_swap(
        self,
        W_src: Tensor,
        target_shape: tuple,
        mask: Optional[Tensor]
    ) -> Tensor:
        assert W_src.shape == target_shape, \
            f"Direct swap requires identical shapes: {W_src.shape} vs {target_shape}"

        if mask is None:
            return W_src.clone().detach()

        # Only swap masked elements, initialize rest with Xavier
        W_target = self._xavier_init(target_shape, W_src.dtype)
        W_target[mask] = W_src[mask]
        return W_target

    def _projected_swap(
        self,
        W_src: Tensor,
        target_shape: tuple,
        mask: Optional[Tensor]
    ) -> Tensor:
        """Project source weight to target shape via SVD truncation."""
        W_2d = W_src.reshape(W_src.shape[0], -1).float()
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

        d_out_tgt, d_in_tgt = target_shape[0], int(torch.tensor(target_shape[1:]).prod().item())
        rank = min(len(S), d_out_tgt, d_in_tgt)

        U_r  = U[:d_out_tgt, :rank]
        S_r  = S[:rank]
        Vh_r = Vh[:rank, :d_in_tgt]

        W_projected = (U_r * S_r.unsqueeze(0)) @ Vh_r
        return W_projected.reshape(target_shape).to(W_src.dtype)

    def _xavier_init(self, shape: tuple, dtype: torch.dtype) -> Tensor:
        W = torch.empty(shape, dtype=dtype)
        torch.nn.init.xavier_uniform_(W if W.dim() >= 2 else W.unsqueeze(0))
        return W


# ─────────────────────────────────────────────
# MAIN PARADOM ENTRY POINT
# ─────────────────────────────────────────────

class Paradom:
    """
    Main entry point for the Paradom framework.

    Usage:
        engine = Paradom()
        result = engine.swap(
            source="meta-llama/Llama-3-8B",
            target=TargetSpec.from_yaml("configs/mamba_7b.yaml"),
            config=SwapConfig(swap_fraction=0.20),
            output_path="./output/llama_as_mamba"
        )
        print(f"Quality: {result.quality_tier.value}")
    """

    def __init__(self):
        self.loader      = ModelLoader()
        self.role_matcher = FunctionalRoleMatcher()
        self.scorer      = ImportanceScorer()
        self.swap_engine = SwapEngine()

    def swap(
        self,
        source: str,
        target: "TargetSpec",
        config: "SwapConfig",
        output_path: str
    ) -> SwapValidationReport:
        """
        Run the full swap pipeline:
        Load → Assign Roles → Score → Identify Equivalences → Swap → Validate → Save

        Memory: Streaming — one layer at a time.
        """
        from pathlib import Path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        equivalence_map = self.identify(source, target)
        swap_results = {}

        for weight in self.loader.stream_layers(source):
            # Find the equivalent pair for this weight
            pair = self._find_pair(equivalence_map, weight.name)

            if pair is None or not pair.is_safe_to_swap:
                # No equivalent — initialize fresh
                swap_results[pair.target_layer_name] = \
                    self.swap_engine.swap(weight.tensor, pair.target_shape, SwapType.SKIP)
                continue

            # Score importance
            mask = self.scorer.score_svd_spectrum(
                weight.tensor,
                top_k_fraction=config.swap_fraction
            )

            # Swap
            swapped = self.swap_engine.swap(
                weight.tensor,
                pair.target_shape,
                pair.swap_type,
                importance_mask=mask if config.swap_fraction < 1.0 else None
            )
            swap_results[pair.target_layer_name] = swapped

        # Save output
        self._save(swap_results, target, output_path)

        # Validate if requested
        if config.validate_after:
            return self._validate(source, str(output_path), target.paradigm)

        return self._build_report(swap_results, equivalence_map)

    def identify(
        self,
        source: str,
        target: "TargetSpec"
    ) -> EquivalenceMap:
        """
        Run equivalence identification only — without performing the swap.
        Useful for previewing what would be swapped before committing.
        """
        ...

    def _find_pair(
        self,
        equivalence_map: EquivalenceMap,
        layer_name: str
    ) -> Optional[EquivalencePair]:
        for pair in equivalence_map.pairs:
            if pair.source.name == layer_name:
                return pair
        return None

    def _save(
        self,
        weights: Dict[str, Tensor],
        target: "TargetSpec",
        output_path: Path
    ):
        from safetensors.torch import save_file
        save_file(weights, output_path / "model.safetensors")
        target.config.save(output_path / "config.json")

    def _validate(
        self,
        source: str,
        converted_path: str,
        paradigm: str
    ) -> SwapValidationReport:
        ...

    def _build_report(
        self,
        swap_results: Dict,
        equivalence_map: EquivalenceMap
    ) -> SwapValidationReport:
        ...
