import torch
from torch import Tensor
from typing import Dict, List, Tuple, Any, Optional
from paradom.core.enums import SwapType, FunctionalRole
from paradom.core.types import EquivalencePair, EquivalenceMap, WeightProduct
from paradom.core.cka import weight_cka

class TransformerToTransformerMapper:
    """
    Standard Phase 2 Mapper: Full-scale Transformer to Transformer Arch-morphing.
    Scales params up/down dynamically (e.g. 8B -> 70B).
    Uses DIRECT swap when shapes are identical for maximum fidelity.
    """

    def __init__(self, swap_engine=None, scorer=None, force_projected=False):
        from paradom.core.swap_engine import SwapEngine
        from paradom.core.importance import ImportanceScorer
        self.swap_engine = swap_engine or SwapEngine()
        self.scorer      = scorer      or ImportanceScorer()
        self.force_projected = force_projected

    def convert(
        self,
        source_products: List[WeightProduct],
        target_config: Dict[str, Any],
        swap_fraction: float = 1.0
    ) -> Tuple[Dict[str, Tensor], EquivalenceMap]:
        """
        Maps a list of Transformer WeightProducts to another Transformer's state_dict.
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

        # 1. Global Layers
        if FunctionalRole.EMBEDDING in global_params:
            wp = global_params[FunctionalRole.EMBEDDING]
            v_size = target_config.get("vocab_size", wp.shape[0])
            d_mod = target_config.get("d_model", wp.shape[1])
            target["embed_tokens.weight"] = self._apply_swap(wp, (v_size, d_mod), swap_fraction, pairs, cka_scores, "embed_tokens.weight", axis_labels=('vocab', 'd_model'))
        
        if FunctionalRole.OUTPUT_HEAD in global_params:
            wp = global_params[FunctionalRole.OUTPUT_HEAD]
            v_size = target_config.get("vocab_size", wp.shape[0])
            d_mod = target_config.get("d_model", wp.shape[1])
            target["lm_head.weight"] = self._apply_swap(wp, (v_size, d_mod), swap_fraction, pairs, cka_scores, "lm_head.weight", axis_labels=('vocab', 'd_model'))

        if FunctionalRole.FINAL_NORMALIZATION in global_params:
            wp = global_params[FunctionalRole.FINAL_NORMALIZATION]
            d_mod = target_config.get("d_model", wp.shape[0])
            target["norm.weight"] = self._apply_swap(wp, (d_mod,), swap_fraction, pairs, cka_scores, "norm.weight", axis_labels=('d_model',))

        # 2. Iterate through Layers
        d_model = target_config["d_model"]
        d_inner = target_config.get("d_inner", d_model * 4) 
        
        # For Transformer-to-Transformer, map sequentially. If target has more layers, we can loop back or pad.
        num_target_layers = target_config.get("num_hidden_layers", len(layers))
        source_layer_keys = sorted(layers.keys())

        for i in range(num_target_layers):
            # Source layer wrapping if target is deeper
            src_i = source_layer_keys[i % len(source_layer_keys)]
            l_src = layers[src_i]
            
            # Map Norms
            if FunctionalRole.NORMALIZATION in l_src: 
                wp = l_src[FunctionalRole.NORMALIZATION]
                target[f"layers.{i}.input_layernorm.weight"] = self._apply_swap(wp, (d_model,), swap_fraction, pairs, cka_scores, f"layers.{i}.input_layernorm.weight", axis_labels=('d_model',))

            if FunctionalRole.POST_NORMALIZATION in l_src:
                wp = l_src[FunctionalRole.POST_NORMALIZATION]
                target[f"layers.{i}.post_attention_layernorm.weight"] = self._apply_swap(wp, (d_model,), swap_fraction, pairs, cka_scores, f"layers.{i}.post_attention_layernorm.weight", axis_labels=('d_model',))

            # Map Attention (Q, K, V, O)
            num_heads = target_config.get("num_heads", 32)
            num_kv = target_config.get("num_key_value_heads", 8)
            head_dim = target_config.get("head_dim", d_model // num_heads)
            for role, t_name, t_shape, axes in [
                (FunctionalRole.CONTEXT_QUERY, "q_proj", num_heads * head_dim, ('q_heads', 'd_model')),
                (FunctionalRole.CONTEXT_KEY, "k_proj", num_kv * head_dim, ('k_heads', 'd_model')),
                (FunctionalRole.CONTEXT_VALUE, "v_proj", num_kv * head_dim, ('v_heads', 'd_model')),
            ]:
                if role in l_src:
                    wp = l_src[role]
                    target[f"layers.{i}.self_attn.{t_name}.weight"] = self._apply_swap(wp, (t_shape, d_model), swap_fraction, pairs, cka_scores, f"layers.{i}.self_attn.{t_name}.weight", axis_labels=axes)

            if FunctionalRole.CONTEXT_OUTPUT in l_src:
                wp = l_src[FunctionalRole.CONTEXT_OUTPUT]
                o_dim = num_heads * head_dim
                target[f"layers.{i}.self_attn.o_proj.weight"] = self._apply_swap(wp, (d_model, o_dim), swap_fraction, pairs, cka_scores, f"layers.{i}.self_attn.o_proj.weight", axis_labels=('d_model', 'o_input'))

            # FFN — gate_proj, up_proj, down_proj
            if FunctionalRole.FFN_GATE in l_src:
                wp_gate = l_src[FunctionalRole.FFN_GATE]
                target[f"layers.{i}.mlp.gate_proj.weight"] = self._apply_swap(wp_gate, (d_inner, d_model), swap_fraction, pairs, cka_scores, f"layers.{i}.mlp.gate_proj.weight", axis_labels=('d_inner', 'd_model'))

            if FunctionalRole.FFN_EXPAND in l_src:
                wp_up = l_src[FunctionalRole.FFN_EXPAND]
                target[f"layers.{i}.mlp.up_proj.weight"] = self._apply_swap(wp_up, (d_inner, d_model), swap_fraction, pairs, cka_scores, f"layers.{i}.mlp.up_proj.weight", axis_labels=('d_inner', 'd_model'))

            if FunctionalRole.FFN_CONTRACT in l_src:
                wp_down = l_src[FunctionalRole.FFN_CONTRACT]
                target[f"layers.{i}.mlp.down_proj.weight"] = self._apply_swap(wp_down, (d_model, d_inner), swap_fraction, pairs, cka_scores, f"layers.{i}.mlp.down_proj.weight", axis_labels=('d_model', 'd_inner'))

        mean_cka = sum(cka_scores.values()) / max(len(cka_scores), 1)
        return target, EquivalenceMap(
            source_model="TransformerSource",
            target_architecture="TransformerTarget",
            pairs=pairs,
            unmapped_source=[],
            uninitialized_target=[],
            mean_cka=mean_cka,
            estimated_quality_tier="acceptable" if mean_cka > 0.50 else "degraded"
        )

    def _apply_swap(self, wp, target_shape, fraction, pairs, scores, target_name=None, axis_labels=None):
        target_name = target_name or wp.name
        
        # If shapes match AND we're not forcing projection AND full fraction → direct copy
        if not self.force_projected and (tuple(wp.tensor.shape) == tuple(target_shape)):
            if fraction >= 1.0:
                out = wp.tensor.clone().detach()
                swap_type = SwapType.DIRECT
            else:
                mask = self.scorer.score_svd_spectrum(wp.tensor, fraction)
                out = self.swap_engine.swap(wp.tensor, target_shape, SwapType.DIRECT, mask, axis_labels=axis_labels)
                swap_type = SwapType.DIRECT
        else:
            # Dynamically route: Condensation (Downscale) uses OT, Expansion (Upscale) uses SVD
            is_downscale = wp.tensor.dim() >= 1 and (
                wp.tensor.shape[0] > target_shape[0] or 
                (wp.tensor.dim() > 1 and wp.tensor.shape[1] > target_shape[1])
            )
            
            swap_type = SwapType.OT if is_downscale else SwapType.PROJECTED
            
            mask = self.scorer.score_svd_spectrum(wp.tensor, fraction) if fraction < 1.0 else None
            out = self.swap_engine.swap(wp.tensor, target_shape, swap_type, mask, axis_labels=axis_labels)
        
        cka = weight_cka(wp.tensor, out)
        pairs.append(EquivalencePair(wp, target_name, target_shape, cka_score=cka, swap_type=swap_type, confidence=1.0))
        scores[target_name] = cka
        return out
