import torch
from torch import Tensor
from typing import Dict, List, Tuple, Any, Optional
from paradom.core.enums import SwapType, FunctionalRole
from paradom.core.types import EquivalencePair, EquivalenceMap, WeightProduct
from paradom.core.cka import weight_cka
from paradom.core.subspace import SubspaceProjector, apply_subspace_projectors

class TransformerToTransformerMapper:
    """
    Standard Phase 2 Mapper: Full-scale Transformer to Transformer Arch-morphing.
    Scales params up/down dynamically (e.g. 8B -> 70B).
    Uses DIRECT swap when shapes are identical for maximum fidelity.
    """

    def __init__(self, swap_engine=None, scorer=None, force_projected=False, projection_method="svd"):
        from paradom.core.swap_engine import SwapEngine
        from paradom.core.importance import ImportanceScorer
        self.swap_engine = swap_engine or SwapEngine()
        self.scorer      = scorer      or ImportanceScorer()
        self.force_projected = force_projected
        self.projection_method = projection_method
        self.subspace = SubspaceProjector()

    def convert(
        self,
        source_products: List[WeightProduct],
        target_config: Dict[str, Any],
        swap_fraction: float = 1.0,
        calibration_data: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], EquivalenceMap]:
        """
        Maps a list of Transformer WeightProducts to another Transformer's state_dict.
        """
        self.subspace.compute(source_products, target_config, method=self.projection_method, calibration_data=calibration_data)

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
            for role, t_name, t_shape, axes, tgt_n_heads in [
                (FunctionalRole.CONTEXT_QUERY, "q_proj", num_heads * head_dim, ('q_heads', 'd_model'), num_heads),
                (FunctionalRole.CONTEXT_KEY, "k_proj", num_kv * head_dim, ('k_heads', 'd_model'), num_kv),
                (FunctionalRole.CONTEXT_VALUE, "v_proj", num_kv * head_dim, ('v_heads', 'd_model'), num_kv),
            ]:
                if role in l_src:
                    wp = l_src[role]
                    src_n_heads = wp.shape[0] // head_dim
                    use_head_aware = (
                        src_n_heads > tgt_n_heads
                        and wp.shape[0] > 0
                        and wp.shape[0] % head_dim == 0
                    )
                    if use_head_aware:
                        out = self._head_aware_attention_swap(
                            wp, (t_shape, d_model), swap_fraction, pairs, cka_scores,
                            f"layers.{i}.self_attn.{t_name}.weight", axes,
                            src_n_heads, tgt_n_heads, head_dim, head_axis=0
                        )
                    else:
                        out = self._apply_swap(wp, (t_shape, d_model), swap_fraction, pairs, cka_scores,
                                               f"layers.{i}.self_attn.{t_name}.weight", axes)
                    target[f"layers.{i}.self_attn.{t_name}.weight"] = out

            if FunctionalRole.CONTEXT_OUTPUT in l_src:
                wp = l_src[FunctionalRole.CONTEXT_OUTPUT]
                o_dim = num_heads * head_dim
                src_n_heads = wp.shape[1] // head_dim
                use_head_aware = (
                    src_n_heads > num_heads
                    and wp.shape[1] > 0
                    and wp.shape[1] % head_dim == 0
                )
                if use_head_aware:
                    out = self._head_aware_attention_swap(
                        wp, (d_model, o_dim), swap_fraction, pairs, cka_scores,
                        f"layers.{i}.self_attn.o_proj.weight", ('d_model', 'o_input'),
                        src_n_heads, num_heads, head_dim, head_axis=1
                    )
                else:
                    out = self._apply_swap(wp, (d_model, o_dim), swap_fraction, pairs, cka_scores,
                                           f"layers.{i}.self_attn.o_proj.weight", ('d_model', 'o_input'))
                target[f"layers.{i}.self_attn.o_proj.weight"] = out

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

        W_projected = apply_subspace_projectors(wp, self.subspace)
        projected_shape = tuple(W_projected.shape)

        if not self.force_projected and projected_shape == target_shape:
            if fraction >= 1.0:
                out = W_projected.clone().detach()
                swap_type = SwapType.DIRECT
            else:
                mask = self.scorer.score_svd_spectrum(W_projected, fraction)
                out = self.swap_engine.swap(W_projected, target_shape, SwapType.DIRECT, mask, axis_labels=axis_labels)
                swap_type = SwapType.DIRECT
        else:
            is_downscale = W_projected.dim() >= 1 and (
                W_projected.shape[0] > target_shape[0] or
                (W_projected.dim() > 1 and W_projected.shape[1] > target_shape[1])
            )

            swap_type = SwapType.OT if is_downscale else SwapType.PROJECTED

            mask = self.scorer.score_svd_spectrum(W_projected, fraction) if fraction < 1.0 else None
            out = self.swap_engine.swap(W_projected, target_shape, swap_type, mask, axis_labels=axis_labels)

        cka = weight_cka(wp.tensor, out)
        pairs.append(EquivalencePair(wp, target_name, target_shape, cka_score=cka, swap_type=swap_type, confidence=1.0))
        scores[target_name] = cka
        return out

    def _head_aware_attention_swap(
        self, wp, target_shape, fraction, pairs, scores, target_name, axis_labels,
        src_n_heads, tgt_n_heads, head_dim, head_axis
    ):
        W_src = wp.tensor
        W_original = W_src.clone()
        modified = W_src.clone()
        src_shape = W_src.shape

        if src_n_heads != tgt_n_heads:
            if head_axis == 0:
                M = src_shape[1]
                W_3d = modified.reshape(src_n_heads, head_dim, M)
                norms = torch.norm(W_3d.reshape(src_n_heads, -1), dim=1)
                _, keep = torch.topk(norms, tgt_n_heads)
                keep = torch.sort(keep)[0]
                modified = W_3d[keep].reshape(tgt_n_heads * head_dim, M)
            else:
                M = src_shape[0]
                W_3d = modified.reshape(M, src_n_heads, head_dim)
                norms = torch.sqrt(torch.sum(W_3d ** 2, dim=(0, 2)))
                _, keep = torch.topk(norms, tgt_n_heads)
                keep = torch.sort(keep)[0]
                modified = W_3d[:, keep].reshape(M, tgt_n_heads * head_dim)

        if self.subspace.P_dmodel is not None:
            if head_axis == 0:
                modified = modified @ self.subspace.P_dmodel
            else:
                modified = self.subspace.P_dmodel.T @ modified

        modified_shape = tuple(modified.shape)

        if modified_shape == target_shape:
            out = modified.clone().detach()
            swap_type = SwapType.DIRECT
        else:
            is_downscale = modified.dim() >= 1 and (
                modified.shape[0] > target_shape[0] or
                (modified.dim() > 1 and modified.shape[1] > target_shape[1])
            )
            swap_type = SwapType.OT if is_downscale else SwapType.PROJECTED
            mask = self.scorer.score_svd_spectrum(modified, fraction) if fraction < 1.0 else None
            out = self.swap_engine.swap(modified, target_shape, swap_type, mask, axis_labels=axis_labels)

        cka = weight_cka(W_original, out)
        pairs.append(EquivalencePair(wp, target_name, target_shape, cka_score=cka, swap_type=swap_type, confidence=1.0))
        scores[target_name] = cka
        return out
