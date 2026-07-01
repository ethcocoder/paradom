import torch
from pathlib import Path
from typing import Dict, Optional, List, Any
from .loader import ModelLoader
from .matcher import FunctionalRoleMatcher
from .importance import ImportanceScorer
from .swap_engine import SwapEngine
from .enums import SwapType, FunctionalRole, QualityTier
from .types import (
    WeightProduct, 
    EquivalencePair, 
    EquivalenceMap, 
    SwapValidationReport
)

class Paradom:
    """
    Main entry point for the Paradom framework.
    """

    def __init__(self):
        self.role_matcher = FunctionalRoleMatcher()
        self.loader      = ModelLoader(role_matcher=self.role_matcher)
        self.scorer      = ImportanceScorer()
        self.swap_engine = SwapEngine()

    def swap(
        self,
        source: str,
        target_architecture: str,
        target_config: Dict[str, Any],
        swap_fraction: float = 0.20,
        output_path: Optional[str] = None
    ) -> SwapValidationReport:
        """
        Run the full swap pipeline:
        Load → Assign Roles → Identify Equivalences → Swap → Save
        """
        if output_path:
            out_p = Path(output_path)
            out_p.mkdir(parents=True, exist_ok=True)

        # 1. Identify Equivalences (Simplified for Phase 1)
        # In this phase, we assume a one-to-one mapping if roles match.
        equivalence_map = self.identify(source, target_architecture, target_config)
        
        swap_results = {}
        total_weights = 0
        swapped_weights = 0
        
        # 2. Iterate and Swap
        # Note: In streaming mode, we'd iterate over source.
        for weight in self.loader.stream_layers(source, architecture="llama"): # Source default for P1
            total_weights += 1
            pair = self._find_pair(equivalence_map, weight.name)

            if pair and pair.is_safe_to_swap:
                # Score importance
                mask = self.scorer.score_svd_spectrum(
                    weight.tensor,
                    top_k_fraction=swap_fraction
                )

                # Execute swap
                swapped = self.swap_engine.swap(
                    weight.tensor,
                    pair.target_shape,
                    pair.swap_type,
                    importance_mask=mask if swap_fraction < 1.0 else None
                )
                swap_results[pair.target_layer_name] = swapped
                swapped_weights += 1
            else:
                # If no mapping, we'd normally initialize fresh if it's a target requirement
                pass

        # 3. Save
        if output_path:
             from safetensors.torch import save_file
             save_file(swap_results, out_p / "model.safetensors")

        # 4. Generate basic report
        return SwapValidationReport(
            source_model=source if isinstance(source, str) else "in-memory",
            target_architecture=target_architecture,
            source_paradigm="llm",
            target_paradigm="llm",
            total_weights=total_weights,
            weights_swapped=swapped_weights,
            swap_fraction=swap_fraction,
            swap_type_distribution={"direct": 1.0}, # Simplified
            cka_scores={},
            mean_cka=0.8, # Mock for P1
            paradigm_metric_name="perplexity",
            source_paradigm_metric=0.0,
            converted_paradigm_metric=0.0,
            retention_fraction=0.0,
            quality_tier=QualityTier.GOOD,
            recommendation="Phase 1 PoC",
            conversion_time_seconds=0.0,
            peak_ram_mb=0.0
        )

    def identify(
        self,
        source: Any,
        target_architecture: str,
        target_config: Dict[str, Any]
    ) -> EquivalenceMap:
        """
        Identifies pairs of weights between source and target.
        """
        # Simplified logic for Phase 1: Match by functional role and layer index
        pairs = []
        source_weights = list(self.loader.stream_layers(source, architecture="llama")) # Mock 
        
        # This is a placeholder for real mapping logic.
        # For Phase 1 Core Experiment, we will manually define a mapping or use
        # a simple heuristic.
        
        return EquivalenceMap(
            source_model=str(source),
            target_architecture=target_architecture,
            pairs=pairs,
            unmapped_source=[],
            uninitialized_target=[],
            mean_cka=1.0,
            estimated_quality_tier=QualityTier.EXCELLENT
        )

    def _find_pair(
        self,
        equivalence_map: EquivalenceMap,
        layer_name: str
    ) -> Optional[EquivalencePair]:
        for pair in equivalence_map.pairs:
            if pair.source.name == layer_name:
                return pair
        return None
