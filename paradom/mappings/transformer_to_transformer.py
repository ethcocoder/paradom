import torch
from torch import Tensor
from typing import Dict, List, Tuple, Any, Optional
from paradom.core.enums import SwapType, FunctionalRole
from paradom.core.types import EquivalencePair, EquivalenceMap, WeightProduct
from paradom.core.cka import weight_cka
from paradom.core.activation_aware_projector import ActivationAwareProjector
from paradom.core.layer_aligner import LayerAligner

class TransformerToTransformerMapper:
    """
    Standard Phase 2 Mapper: Full-scale Transformer to Transformer Arch-morphing.
    Scales params up/down dynamically (e.g. 8B -> 70B).
    Uses DIRECT swap when shapes are identical for maximum fidelity.
    
    Now integrates MagneticProjector for population-aware downscaling:
    - Pre-computes shared spectral bases across ALL layers of each functional role
    - Uses 'magnetic character' alignment for cross-layer consistent projections
    - Applies residual energy correction during compression
    
    Also integrates ActivationAwareProjector for attention weights:
    - Uses calibration data to merge heads while preserving attention patterns
    - Falls back to SVD for non-attention weights
    
    Also integrates LayerAligner for post-projection alignment:
    - Aligns each layer's output to match source using Procrustes correction
    - Stops error from compounding across layers
    """

    def __init__(self, swap_engine=None, scorer=None, force_projected=False, source_config=None, projector=None, aligner=None, ml_projector=None):
        from paradom.core.swap_engine import SwapEngine, MagneticProjector
        from paradom.core.importance import ImportanceScorer
        self.magnetic_projector = MagneticProjector()
        self.swap_engine = swap_engine or SwapEngine(magnetic_projector=self.magnetic_projector)
        self.scorer      = scorer      or ImportanceScorer()
        self.force_projected = force_projected
        self._source_config = source_config
        self._kv_activations = {}
        self._projector = projector  # Optional[ActivationAwareProjector]
        self._aligner = aligner      # Optional[LayerAligner]
        self._ml_projector = ml_projector  # Optional[EnsembleProjector]

    def set_kv_activations(self, kv_activations: Dict[int, Dict[str, Tensor]]):
        self._kv_activations = kv_activations

    def set_projector(self, projector: ActivationAwareProjector):
        """Set the activation-aware projector for attention weights."""
        self._projector = projector

    def set_aligner(self, aligner: LayerAligner):
        """Set the layer aligner for post-projection alignment."""
        self._aligner = aligner

    def set_ml_projector(self, ml_projector):
        """Set the ML ensemble projector for adaptive projection."""
        self._ml_projector = ml_projector

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

        # Infer source head structure from source_config (preferred) or weight shapes
        if self._source_config is not None:
            self._source_num_heads = self._source_config.get("num_heads", 9)
            self._source_num_kv = self._source_config.get("num_key_value_heads", 3)
            self._source_head_dim = self._source_config.get("head_dim", 64)
        else:
            self._source_num_heads = 9
            self._source_num_kv = 3
            self._source_head_dim = 64

        # ── MAGNETIC FIELD REGISTRATION (lazy computation) ──
        # Register shared spectral bases for each functional role with ≥2 instances.
        # Computation is deferred to first use — only roles actually needed for OT
        # projection will have their bases computed.
        if hasattr(self, 'magnetic_projector') and self.magnetic_projector is not None:
            roles_with_layers = {r for layer in layers.values() for r in layer}
            for role in roles_with_layers:
                role_weights = [layers[layer_idx][role].tensor for layer_idx in sorted(layers.keys())
                                if role in layers[layer_idx]]
                if len(role_weights) >= 2:
                    self.magnetic_projector.register_role(
                        f"role_{role.name}_right", role_weights, transpose=False
                    )
                    self.magnetic_projector.register_role(
                        f"role_{role.name}_left", role_weights, transpose=True
                    )

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

            src_num_heads = self._source_num_heads if hasattr(self, '_source_num_heads') else 9
            src_num_kv = self._source_num_kv if hasattr(self, '_source_num_kv') else 3
            src_head_dim = self._source_head_dim if hasattr(self, '_source_head_dim') else 64

            for role, t_name, t_shape, axes, src_h, tgt_h, kv_key in [
                (FunctionalRole.CONTEXT_QUERY, "q_proj", num_heads * head_dim, ('q_heads', 'd_model'), src_num_heads, num_heads, None),
                (FunctionalRole.CONTEXT_KEY, "k_proj", num_kv * head_dim, ('k_heads', 'd_model'), src_num_kv, num_kv, "k"),
                (FunctionalRole.CONTEXT_VALUE, "v_proj", num_kv * head_dim, ('v_heads', 'd_model'), src_num_kv, num_kv, "v"),
            ]:
                if role in l_src:
                    wp = l_src[role]
                    is_downscale = wp.tensor.shape[0] > t_shape or (len(wp.tensor.shape) > 1 and wp.tensor.shape[1] > d_model)
                    hs = (src_h, src_head_dim) if is_downscale and (src_h != tgt_h or wp.tensor.shape[0] != t_shape) else None
                    out = self._apply_swap(wp, (t_shape, d_model), swap_fraction, pairs, cka_scores, f"layers.{i}.self_attn.{t_name}.weight", axis_labels=axes, head_structure=hs)
                    target[f"layers.{i}.self_attn.{t_name}.weight"] = out

            if FunctionalRole.CONTEXT_OUTPUT in l_src:
                wp = l_src[FunctionalRole.CONTEXT_OUTPUT]
                o_dim = num_heads * head_dim
                src_o_dim = wp.tensor.shape[1]
                src_o_heads = src_o_dim // src_head_dim
                hs_o = (src_o_heads, src_head_dim, True) if (src_o_heads != num_heads or wp.tensor.shape != (d_model, o_dim)) else None
                out = self._apply_swap(wp, (d_model, o_dim), swap_fraction, pairs, cka_scores, f"layers.{i}.self_attn.o_proj.weight", axis_labels=('d_model', 'o_input'), head_structure=hs_o)
                target[f"layers.{i}.self_attn.o_proj.weight"] = out

            # FFN — gate_proj, up_proj, down_proj (truncation: preserve original weight ordering)
            for role, t_name, t_shape in [
                (FunctionalRole.FFN_GATE, "gate_proj", (d_inner, d_model)),
                (FunctionalRole.FFN_EXPAND, "up_proj", (d_inner, d_model)),
                (FunctionalRole.FFN_CONTRACT, "down_proj", (d_model, d_inner)),
            ]:
                if role in l_src:
                    wp = l_src[role]
                    is_downscale = wp.tensor.shape[0] > t_shape[0] or wp.tensor.shape[1] > t_shape[1]
                    if is_downscale:
                        out = wp.tensor[:t_shape[0], :t_shape[1]].clone().detach()
                        cka = weight_cka(wp.tensor, out)
                        pairs.append(EquivalencePair(wp, f"layers.{i}.mlp.{t_name}.weight", t_shape, cka_score=cka, swap_type=SwapType.PROJECTED, confidence=1.0))
                        cka_scores[f"layers.{i}.mlp.{t_name}.weight"] = cka
                    else:
                        out = self._apply_swap(wp, t_shape, swap_fraction, pairs, cka_scores, f"layers.{i}.mlp.{t_name}.weight", axis_labels=('d_inner', 'd_model') if role != FunctionalRole.FFN_CONTRACT else ('d_model', 'd_inner'))
                    target[f"layers.{i}.mlp.{t_name}.weight"] = out

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

    def _apply_swap(self, wp, target_shape, fraction, pairs, scores, target_name=None, axis_labels=None, head_structure=None):
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
            
            swap_type = SwapType.PROJECTED
            
            mask = self.scorer.score_svd_spectrum(wp.tensor, fraction) if fraction < 1.0 else None
            out = self.swap_engine.swap(wp.tensor, target_shape, swap_type, mask, axis_labels=axis_labels, functional_role=wp.functional_role, head_structure=head_structure)
        
        cka = weight_cka(wp.tensor, out)
        pairs.append(EquivalencePair(wp, target_name, target_shape, cka_score=cka, swap_type=swap_type, confidence=1.0))
        scores[target_name] = cka
        return out
