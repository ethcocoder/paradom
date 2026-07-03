import time
import torch
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
from .loader import ModelLoader
from .matcher import FunctionalRoleMatcher
from .importance import ImportanceScorer
from .swap_engine import SwapEngine
from .enums import SwapType, FunctionalRole, QualityTier
from .types import (
    WeightProduct,
    EquivalencePair,
    EquivalenceMap,
    SwapValidationReport,
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
        source: Union[str, Dict[str, Any]],
        target_architecture: str,
        target_config: Dict[str, Any],
        source_architecture: str = "tinytransformer",
        swap_fraction: float = 0.20,
        output_path: Optional[str] = None
    ) -> SwapValidationReport:
        """
        Run the full swap pipeline:
        Load → Assign Roles → Identify Equivalences → Swap → Save
        """
        t0 = time.perf_counter()

        if output_path:
            out_p = Path(output_path)
            out_p.mkdir(parents=True, exist_ok=True)

        target_arch = target_architecture.lower()

        if target_arch == "tinymamba" and source_architecture.lower() == "tinytransformer":
            return self._swap_tiny_transformer_to_mamba(
                source, swap_fraction, output_path, t0
            )

        equivalence_map = self.identify(
            source, target_architecture, target_config, source_architecture
        )

        swap_results = {}
        total_weights = 0
        swapped_weights = 0
        swap_type_counts: Dict[str, int] = {}

        for weight in self.loader.stream_layers(source, architecture=source_architecture):
            total_weights += 1
            pair = self._find_pair(equivalence_map, weight.name)

            if pair and pair.is_safe_to_swap:
                mask = self.scorer.score_svd_spectrum(
                    weight.tensor,
                    top_k_fraction=swap_fraction
                )
                swapped = self.swap_engine.swap(
                    weight.tensor,
                    pair.target_shape,
                    pair.swap_type,
                    importance_mask=mask if swap_fraction < 1.0 else None
                )
                swap_results[pair.target_layer_name] = swapped
                swapped_weights += 1
                key = pair.swap_type.value
                swap_type_counts[key] = swap_type_counts.get(key, 0) + 1

        if output_path:
            from safetensors.torch import save_file
            save_file(swap_results, out_p / "model.safetensors")

        dist = self._swap_distribution(swap_type_counts, swapped_weights)
        elapsed = time.perf_counter() - t0

        return SwapValidationReport(
            source_model=source if isinstance(source, str) else "in-memory",
            target_architecture=target_architecture,
            source_paradigm="llm",
            target_paradigm="llm",
            total_weights=total_weights,
            weights_swapped=swapped_weights,
            swap_fraction=swap_fraction,
            swap_type_distribution=dist,
            cka_scores={p.target_layer_name: p.cka_score for p in equivalence_map.pairs},
            mean_cka=equivalence_map.mean_cka,
            paradigm_metric_name="perplexity",
            source_paradigm_metric=0.0,
            converted_paradigm_metric=0.0,
            retention_fraction=0.0,
            quality_tier=equivalence_map.estimated_quality_tier,
            recommendation="Phase 1 generic role-matched swap",
            conversion_time_seconds=elapsed,
            peak_ram_mb=0.0,
        )

    def _swap_tiny_transformer_to_mamba(
        self,
        source: Union[str, Dict[str, Any]],
        swap_fraction: float,
        output_path: Optional[str],
        t0: float,
    ) -> SwapValidationReport:
        from paradom.mappings.tiny_transformer_to_mamba import TinyTransformerToMambaMapper

        source_dict = self._load_state_dict(source)
        mapper = TinyTransformerToMambaMapper()
        swap_results, equivalence_map = mapper.convert(source_dict, swap_fraction)

        swap_type_counts: Dict[str, int] = {}
        for pair in equivalence_map.pairs:
            key = pair.swap_type.value
            swap_type_counts[key] = swap_type_counts.get(key, 0) + 1

        if output_path:
            from safetensors.torch import save_file
            out_p = Path(output_path)
            save_file(swap_results, out_p / "model.safetensors")
            report_path = out_p / "swap_report.json"
            self._save_report(equivalence_map, swap_results, swap_fraction, report_path)

        elapsed = time.perf_counter() - t0
        swapped = len(swap_results)

        return SwapValidationReport(
            source_model=source if isinstance(source, str) else "in-memory",
            target_architecture="tinymamba",
            source_paradigm="llm",
            target_paradigm="llm",
            total_weights=len(source_dict),
            weights_swapped=swapped,
            swap_fraction=swap_fraction,
            swap_type_distribution=self._swap_distribution(swap_type_counts, swapped),
            cka_scores={p.target_layer_name: p.cka_score for p in equivalence_map.pairs},
            mean_cka=equivalence_map.mean_cka,
            paradigm_metric_name="perplexity",
            source_paradigm_metric=0.0,
            converted_paradigm_metric=0.0,
            retention_fraction=0.0,
            quality_tier=equivalence_map.estimated_quality_tier,
            recommendation=(
                "Full TinyTransformer→TinyMamba map with SSM derivation "
                f"(mean CKA={equivalence_map.mean_cka:.3f})"
            ),
            conversion_time_seconds=elapsed,
            peak_ram_mb=0.0,
        )

    def identify(
        self,
        source: Any,
        target_architecture: str,
        target_config: Dict[str, Any],
        source_architecture: str = "tinytransformer"
    ) -> EquivalenceMap:
        """
        Identifies pairs of weights between source and target.
        Uses explicit TinyTransformer→TinyMamba mapping when applicable.
        """
        if (
            target_architecture.lower() == "tinymamba"
            and source_architecture.lower() == "tinytransformer"
        ):
            from paradom.mappings.tiny_transformer_to_mamba import TinyTransformerToMambaMapper
            source_dict = self._load_state_dict(source)
            _, equivalence_map = TinyTransformerToMambaMapper().convert(
                source_dict, swap_fraction=1.0
            )
            return equivalence_map

        pairs = []
        source_weights = list(
            self.loader.stream_layers(source, architecture=source_architecture)
        )

        target_requirements = [
            ("embedding.weight", (50257, 256), FunctionalRole.EMBEDDING, -1),
            ("lm_head.weight",   (50257, 256), FunctionalRole.OUTPUT_HEAD, -1),
        ]
        for i in range(2):
            target_requirements.extend([
                (f"layers.{i}.in_proj.weight",  (1024, 256), FunctionalRole.CONTEXT_QUERY, i),
                (f"layers.{i}.x_proj.weight",   (48, 512),   FunctionalRole.CONTEXT_KEY,   i),
                (f"layers.{i}.out_proj.weight", (256, 512),  FunctionalRole.CONTEXT_OUTPUT, i),
                (f"layers.{i}.dt_proj.weight",  (512, 16),   FunctionalRole.FFN_EXPAND,    i),
                (f"layers.{i}.norm.weight",     (256,),      FunctionalRole.NORMALIZATION, i),
            ])

        for target_name, target_shape, role, target_idx in target_requirements:
            best_match = None
            for src_weight in source_weights:
                if src_weight.functional_role == role and src_weight.layer_index == target_idx:
                    best_match = src_weight
                    break

            if best_match:
                swap_type = (
                    SwapType.DIRECT
                    if best_match.tensor.shape == target_shape
                    else SwapType.PROJECTED
                )
                pairs.append(EquivalencePair(
                    source=best_match,
                    target_layer_name=target_name,
                    target_shape=target_shape,
                    cka_score=0.5,
                    swap_type=swap_type,
                    confidence=0.7,
                ))

        mean_cka = sum(p.cka_score for p in pairs) / max(len(pairs), 1)
        return EquivalenceMap(
            source_model=str(source),
            target_architecture=target_architecture,
            pairs=pairs,
            unmapped_source=[],
            uninitialized_target=[],
            mean_cka=mean_cka,
            estimated_quality_tier=QualityTier.ACCEPTABLE,
        )

    def _load_state_dict(self, source: Union[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return source
        weights = torch.load(source, map_location="cpu")
        if isinstance(weights, dict) and "model" in weights:
            return weights["model"]
        return weights

    def _swap_distribution(
        self,
        counts: Dict[str, int],
        total: int,
    ) -> Dict[str, float]:
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    def _save_report(
        self,
        equivalence_map: EquivalenceMap,
        swap_results: Dict[str, torch.Tensor],
        swap_fraction: float,
        path: Path,
    ) -> None:
        import json
        payload = {
            "source_model": equivalence_map.source_model,
            "target_architecture": equivalence_map.target_architecture,
            "swap_fraction": swap_fraction,
            "mean_cka": equivalence_map.mean_cka,
            "quality_tier": equivalence_map.estimated_quality_tier.value,
            "layers_swapped": len(swap_results),
            "pairs": [
                {
                    "source": p.source.name,
                    "target": p.target_layer_name,
                    "swap_type": p.swap_type.value,
                    "cka_score": round(p.cka_score, 4),
                    "confidence": p.confidence,
                }
                for p in equivalence_map.pairs
            ],
            "unmapped_source": equivalence_map.unmapped_source,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _find_pair(
        self,
        equivalence_map: EquivalenceMap,
        layer_name: str
    ) -> Optional[EquivalencePair]:
        for pair in equivalence_map.pairs:
            if pair.source.name == layer_name:
                return pair
        return None
