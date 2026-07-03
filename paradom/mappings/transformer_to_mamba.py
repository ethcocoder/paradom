import torch
from torch import Tensor
from typing import Dict, List, Tuple, Any, Optional
from paradom.core.enums import SwapType, FunctionalRole
from paradom.core.types import EquivalencePair, EquivalenceMap, WeightProduct
from paradom.core.ssm_derivation import (
    derive_A_log_from_attention,
    derive_D_from_value_proj,
    derive_conv1d_from_attention
)
from paradom.core.cka import weight_cka

class TransformerToMambaMapper:
    """
    Standard Phase 2 Mapper: Full-scale Transformer to SSM (Mamba).
    Handles real-world models (7B-70B) with streaming awareness.
    """

    def __init__(self, swap_engine=None, scorer=None):
        from paradom.core.swap_engine import SwapEngine
        from paradom.core.importance import ImportanceScorer
        self.swap_engine = swap_engine or SwapEngine()
        self.scorer      = scorer      or ImportanceScorer()

    def convert(
        self,
        source_products: List[WeightProduct],
        target_config: Dict[str, Any],
        swap_fraction: float = 1.0
    ) -> Tuple[Dict[str, Tensor], EquivalenceMap]:
        """
        Maps a list of Transformer WeightProducts to Mamba state_dict.
        """
        target = {}
        pairs = []
        cka_scores = {}
        
        # Group source products by role and layer
        layers: Dict[int, Dict[FunctionalRole, WeightProduct]] = {}
        global_params: Dict[FunctionalRole, WeightProduct] = {}

        for wp in source_products:
            if wp.layer_index >= 0:
                if wp.layer_index not in layers: layers[wp.layer_index] = {}
                layers[wp.layer_index][wp.functional_role] = wp
            else:
                global_params[wp.functional_role] = wp

        # 1. Global Layers (Embedding/Output)
        if FunctionalRole.EMBEDDING in global_params:
            wp = global_params[FunctionalRole.EMBEDDING]
            target["embedding.weight"] = self._apply_swap(wp, wp.shape, swap_fraction, pairs, cka_scores)
        
        if FunctionalRole.OUTPUT_HEAD in global_params:
            wp = global_params[FunctionalRole.OUTPUT_HEAD]
            target["lm_head.weight"] = self._apply_swap(wp, wp.shape, swap_fraction, pairs, cka_scores)

        # 2. Iterate through Layers
        d_model = target_config["d_model"]
        d_inner = target_config.get("d_inner", d_model * 2)
        d_state = target_config.get("d_state", 16)
        d_conv  = target_config.get("d_conv", 4)

        for i in sorted(layers.keys()):
            l_src = layers[i]
            
            # Map Norm
            if FunctionalRole.NORMALIZATION in l_src:
                wp = l_src[FunctionalRole.NORMALIZATION]
                target[f"layers.{i}.norm.weight"] = self._apply_swap(wp, wp.shape, swap_fraction, pairs, cka_scores, f"layers.{i}.norm.weight")

            # Map Inbound Projections (Attention Q/K/V -> Mamba in_proj)
            if all(r in l_src for r in [FunctionalRole.CONTEXT_QUERY, FunctionalRole.CONTEXT_KEY, FunctionalRole.CONTEXT_VALUE]):
                q = l_src[FunctionalRole.CONTEXT_QUERY].tensor
                k = l_src[FunctionalRole.CONTEXT_KEY].tensor
                v = l_src[FunctionalRole.CONTEXT_VALUE].tensor
                
                # Concatenate to match Mamba's expected in_proj structure [x, z]
                mamba_in = torch.randn(2 * d_inner, d_model, device=v.device, dtype=v.dtype)
                mamba_in[:d_inner, :] = self.swap_engine._projected_swap(v, (d_inner, d_model), None)
                mamba_in[d_inner:, :] = self.swap_engine._projected_swap(torch.cat([q, k], dim=0), (d_inner, d_model), None)
                
                target[f"layers.{i}.in_proj.weight"] = mamba_in

            # Map Outbound Projection
            if FunctionalRole.CONTEXT_OUTPUT in l_src:
                wp = l_src[FunctionalRole.CONTEXT_OUTPUT]
                target[f"layers.{i}.out_proj.weight"] = self._apply_swap(wp, (d_model, d_inner), swap_fraction, pairs, cka_scores, f"layers.{i}.out_proj.weight")

            # Derive SSM Parameters
            if FunctionalRole.CONTEXT_QUERY in l_src and FunctionalRole.CONTEXT_KEY in l_src:
                q = l_src[FunctionalRole.CONTEXT_QUERY].tensor
                k = l_src[FunctionalRole.CONTEXT_KEY].tensor
                v = l_src[FunctionalRole.CONTEXT_VALUE].tensor
                
                target[f"layers.{i}.A_log"] = derive_A_log_from_attention(q, k, d_inner, d_state)
                target[f"layers.{i}.D"]     = derive_D_from_value_proj(v, d_inner)
                target[f"layers.{i}.conv1d.weight"] = derive_conv1d_from_attention(v, d_inner, d_conv)

        mean_cka = sum(cka_scores.values()) / max(len(cka_scores), 1)
        return target, EquivalenceMap(
            source_model="Transformer-7B+",
            target_architecture="Mamba-7B+",
            pairs=pairs,
            unmapped_source=[],
            uninitialized_target=[],
            mean_cka=mean_cka,
            estimated_quality_tier="acceptable"
        )

    def _apply_swap(self, wp, target_shape, fraction, pairs, scores, target_name=None):
        target_name = target_name or wp.name
        mask = self.scorer.score_svd_spectrum(wp.tensor, fraction)
        out = self.swap_engine.swap(wp.tensor, target_shape, SwapType.PROJECTED, mask)
        pairs.append(EquivalencePair(wp, target_name, target_shape, cka_score=weight_cka(wp.tensor, out), swap_type=SwapType.PROJECTED))
        scores[target_name] = pairs[-1].cka_score
        return out
