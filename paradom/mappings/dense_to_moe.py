import torch
from torch import Tensor
from typing import Dict, List, Tuple, Any, Optional
from paradom.core.enums import SwapType, FunctionalRole
from paradom.core.types import EquivalencePair, EquivalenceMap, WeightProduct
from paradom.core.cka import weight_cka

class DenseToMoEMapper:
    """
    Standard Phase 2 Mapper: Full-scale Transformer Dense FFN to MoE.
    Handles architectural transformation for creating sparsely activated experts.
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
        Maps a list of Dense Transformer WeightProducts to MoE state_dict.
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
            v_size = target_config.get("vocab_size", wp.shape[0])
            d_mod = target_config.get("d_model", wp.shape[1])
            target["embed_tokens.weight"] = self._apply_swap(wp, (v_size, d_mod), swap_fraction, pairs, cka_scores, "embed_tokens.weight")
        
        if FunctionalRole.OUTPUT_HEAD in global_params:
            wp = global_params[FunctionalRole.OUTPUT_HEAD]
            v_size = target_config.get("vocab_size", wp.shape[0])
            d_mod = target_config.get("d_model", wp.shape[1])
            target["lm_head.weight"] = self._apply_swap(wp, (v_size, d_mod), swap_fraction, pairs, cka_scores, "lm_head.weight")

        # 2. Iterate through Layers
        d_model = target_config["d_model"]
        d_inner = target_config.get("d_inner", d_model * 4) # Interm size per expert
        n_experts = target_config.get("num_experts", 8)

        for i in sorted(layers.keys()):
            l_src = layers[i]
            
            # Map Norms
            if FunctionalRole.NORMALIZATION in l_src: # Assuming input_layernorm
                wp = l_src[FunctionalRole.NORMALIZATION]
                target[f"layers.{i}.input_layernorm.weight"] = self._apply_swap(wp, (d_model,), swap_fraction, pairs, cka_scores, f"layers.{i}.input_layernorm.weight")

            # Map Attention (Q, K, V, O)
            num_heads = target_config.get("num_heads", 32)
            num_kv = target_config.get("num_key_value_heads", 8)
            head_dim = target_config.get("head_dim", d_model // num_heads)
            for role, t_name, t_shape in [
                (FunctionalRole.CONTEXT_QUERY, "q_proj", num_heads * head_dim),
                (FunctionalRole.CONTEXT_KEY, "k_proj", num_kv * head_dim),
                (FunctionalRole.CONTEXT_VALUE, "v_proj", num_kv * head_dim),
            ]:
                if role in l_src:
                    wp = l_src[role]
                    target[f"layers.{i}.self_attn.{t_name}.weight"] = self._apply_swap(wp, (t_shape, d_model), swap_fraction, pairs, cka_scores, f"layers.{i}.self_attn.{t_name}.weight")

            if FunctionalRole.CONTEXT_OUTPUT in l_src:
                wp = l_src[FunctionalRole.CONTEXT_OUTPUT]
                o_dim = num_heads * head_dim
                target[f"layers.{i}.self_attn.o_proj.weight"] = self._apply_swap(wp, (d_model, o_dim), swap_fraction, pairs, cka_scores, f"layers.{i}.self_attn.o_proj.weight")

            # MoE Routing Gate
            # We initialize to small normal random variable to allow rapid routing divergence.
            dev = l_src[FunctionalRole.FFN_EXPAND].tensor.device if FunctionalRole.FFN_EXPAND in l_src else torch.device("cpu")
            dtp = l_src[FunctionalRole.FFN_EXPAND].tensor.dtype if FunctionalRole.FFN_EXPAND in l_src else torch.float16
            target[f"layers.{i}.block_sparse_moe.gate.weight"] = torch.randn(n_experts, d_model, device=dev, dtype=dtp) * 0.02

            # FFN -> Experts
            if FunctionalRole.FFN_EXPAND in l_src:
                wp_up = l_src[FunctionalRole.FFN_EXPAND]
                base_up = self._apply_swap(wp_up, (d_inner, d_model), swap_fraction, pairs, cka_scores, f"layers.{i}.base_ffn_expand")
                for e in range(n_experts):
                    target[f"layers.{i}.block_sparse_moe.experts.{e}.w1.weight"] = base_up.clone()

            if FunctionalRole.FFN_CONTRACT in l_src:
                wp_down = l_src[FunctionalRole.FFN_CONTRACT]
                base_down = self._apply_swap(wp_down, (d_model, d_inner), swap_fraction, pairs, cka_scores, f"layers.{i}.base_ffn_contract")
                for e in range(n_experts):
                    target[f"layers.{i}.block_sparse_moe.experts.{e}.w2.weight"] = base_down.clone()

        mean_cka = sum(cka_scores.values()) / max(len(cka_scores), 1)
        return target, EquivalenceMap(
            source_model="DenseTransformer",
            target_architecture="MoEType",
            pairs=pairs,
            unmapped_source=[],
            uninitialized_target=[],
            mean_cka=mean_cka,
            estimated_quality_tier="acceptable" if mean_cka > 0.50 else "degraded"
        )

    def _apply_swap(self, wp, target_shape, fraction, pairs, scores, target_name=None):
        target_name = target_name or wp.name
        mask = self.scorer.score_svd_spectrum(wp.tensor, fraction)
        out = self.swap_engine.swap(wp.tensor, target_shape, SwapType.PROJECTED, mask)
        # Avoid creating pairs array loops for cloned experts, just register the base
        if "base_ffn" not in target_name:
            pairs.append(EquivalencePair(wp, target_name, target_shape, cka_score=weight_cka(wp.tensor, out), swap_type=SwapType.PROJECTED, confidence=1.0))
            scores[target_name] = pairs[-1].cka_score
        return out
