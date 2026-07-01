from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import Tensor
from .enums import FunctionalRole, SwapType, QualityTier

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
