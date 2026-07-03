"""
Explicit Transformer → TinyMamba mapping for Phase 1 proof.

Layer correspondence follows docs/specs/SPECIFICATIONS.md §5.1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from paradom.core.cka import weight_cka
from paradom.core.enums import FunctionalRole, QualityTier, SwapType
from paradom.core.importance import ImportanceScorer
from paradom.core.ssm_derivation import (
    derive_A_log_from_attention,
    derive_conv1d_from_attention,
    derive_D_from_value_proj,
)
from paradom.core.swap_engine import SwapEngine
from paradom.core.types import EquivalenceMap, EquivalencePair, WeightProduct


class TinyTransformerToMambaMapper:
    """
    Converts TinyTransformer checkpoints into TinyMamba weight dictionaries.

    Uses direct swaps where shapes align, SVD projection otherwise, and
    mathematical derivation for SSM-only parameters (A_log, D, conv1d).
    """

    N_LAYERS = 2
    VOCAB_SIZE = 50257
    D_MODEL = 256
    D_INNER = 512
    D_STATE = 16
    D_CONV = 4
    DT_RANK = 16

    def __init__(self) -> None:
        self.swap_engine = SwapEngine()
        self.scorer = ImportanceScorer()

    def convert(
        self,
        source: Dict[str, Tensor],
        swap_fraction: float = 0.20,
    ) -> Tuple[Dict[str, Tensor], EquivalenceMap]:
        """Build full TinyMamba state dict from TinyTransformer weights."""
        target: Dict[str, Tensor] = {}
        pairs: List[EquivalencePair] = []
        cka_scores: Dict[str, float] = {}

        target["embedding.weight"] = self._direct(
            source, "embedding.weight", pairs, cka_scores, swap_fraction
        )
        target["lm_head.weight"] = self._direct(
            source, "lm_head.weight", pairs, cka_scores, swap_fraction
        )
        target["norm.weight"] = self._norm_from_layernorm(
            source, "norm.weight", "norm.weight", pairs, cka_scores, swap_fraction
        )

        for i in range(self.N_LAYERS):
            prefix = f"layers.{i}."
            q = source[f"{prefix}q_proj.weight"]
            k = source[f"{prefix}k_proj.weight"]
            v = source[f"{prefix}v_proj.weight"]
            # Content (v) -> SSM branch (x)
            # Selection (q, k) -> Gating branch (z)
            in_first = torch.cat([v, v], dim=0) # (512, 256) -> d_inner
            in_second = torch.cat([q, k], dim=0) # (512, 256) -> d_inner
            in_proj = torch.cat([in_first, in_second], dim=0) # (1024, 256) -> 2*d_inner
            
            in_proj = self._apply_fraction(in_proj, swap_fraction)
            target[f"{prefix}in_proj.weight"] = in_proj
            pairs.append(self._pair(
                f"{prefix}v_proj+q_proj+k_proj",
                f"{prefix}in_proj.weight",
                in_proj.shape,
                torch.cat([v, q, k], dim=0),
                in_proj,
                SwapType.PROJECTED,
            ))
            cka_scores[f"{prefix}in_proj.weight"] = weight_cka(
                torch.cat([v, q, k], dim=0), in_proj
            )
            o = source[f"{prefix}o_proj.weight"]
            out_proj = self._projected(
                o, (self.D_MODEL, self.D_INNER), swap_fraction
            )
            target[f"{prefix}out_proj.weight"] = out_proj
            pairs.append(self._pair(
                f"{prefix}o_proj.weight",
                f"{prefix}out_proj.weight",
                out_proj.shape,
                o,
                out_proj,
                SwapType.PROJECTED,
            ))
            cka_scores[f"{prefix}out_proj.weight"] = weight_cka(o, out_proj)

            gate = source[f"{prefix}gate_proj.weight"]
            up = source[f"{prefix}up_proj.weight"]
            gate_up = torch.cat([gate, up], dim=0)
            x_proj = self._projected(
                gate_up, (self.DT_RANK + 2 * self.D_STATE, self.D_INNER), swap_fraction
            )
            target[f"{prefix}x_proj.weight"] = x_proj
            pairs.append(self._pair(
                f"{prefix}gate_proj+up_proj",
                f"{prefix}x_proj.weight",
                x_proj.shape,
                gate_up,
                x_proj,
                SwapType.PROJECTED,
            ))
            cka_scores[f"{prefix}x_proj.weight"] = weight_cka(gate_up, x_proj)

            down = source[f"{prefix}down_proj.weight"]
            dt_weight = self._projected(
                down, (self.D_INNER, self.DT_RANK), swap_fraction
            )
            target[f"{prefix}dt_proj.weight"] = dt_weight
            dt_bias = down.float().mean(dim=1)
            if dt_bias.numel() != self.D_INNER:
                dt_bias = torch.nn.functional.interpolate(
                    dt_bias.unsqueeze(0).unsqueeze(0),
                    size=self.D_INNER,
                    mode="linear",
                    align_corners=True,
                ).squeeze()
            target[f"{prefix}dt_proj.bias"] = dt_bias.to(down.dtype)
            pairs.append(self._pair(
                f"{prefix}down_proj.weight",
                f"{prefix}dt_proj.weight",
                dt_weight.shape,
                down,
                dt_weight,
                SwapType.PROJECTED,
            ))
            cka_scores[f"{prefix}dt_proj.weight"] = weight_cka(down, dt_weight)

            norm_w = source[f"{prefix}input_layernorm.weight"]
            norm = self._apply_fraction(norm_w.clone(), swap_fraction)
            target[f"{prefix}norm.weight"] = norm
            pairs.append(self._pair(
                f"{prefix}input_layernorm.weight",
                f"{prefix}norm.weight",
                norm.shape,
                norm_w,
                norm,
                SwapType.DIRECT,
            ))
            cka_scores[f"{prefix}norm.weight"] = weight_cka(norm_w, norm)

            A_log = derive_A_log_from_attention(q, k, self.D_INNER, self.D_STATE)
            target[f"{prefix}A_log"] = A_log
            pairs.append(self._pair(
                f"{prefix}q_proj+k_proj (derived)",
                f"{prefix}A_log",
                A_log.shape,
                q,
                A_log,
                SwapType.DERIVED,
                confidence=0.75,
            ))
            cka_scores[f"{prefix}A_log"] = weight_cka(q @ k.T, A_log)

            D = derive_D_from_value_proj(v, self.D_INNER)
            target[f"{prefix}D"] = D
            pairs.append(self._pair(
                f"{prefix}v_proj (derived)",
                f"{prefix}D",
                D.shape,
                v,
                D,
                SwapType.DERIVED,
                confidence=0.70,
            ))
            cka_scores[f"{prefix}D"] = weight_cka(v, D.unsqueeze(1).expand(-1, v.shape[1]))

            conv = derive_conv1d_from_attention(v, self.D_INNER, self.D_CONV)
            target[f"{prefix}conv1d.weight"] = conv
            pairs.append(self._pair(
                f"{prefix}v_proj (derived)",
                f"{prefix}conv1d.weight",
                conv.shape,
                v,
                conv.squeeze(2),
                SwapType.DERIVED,
                confidence=0.65,
            ))
            cka_scores[f"{prefix}conv1d.weight"] = weight_cka(v, conv[:, 0, :])

        mean_cka = sum(cka_scores.values()) / max(len(cka_scores), 1)
        tier = self._quality_tier(mean_cka)

        equivalence_map = EquivalenceMap(
            source_model="tinytransformer",
            target_architecture="tinymamba",
            pairs=pairs,
            unmapped_source=self._unmapped(source, pairs),
            uninitialized_target=[],
            mean_cka=mean_cka,
            estimated_quality_tier=tier,
        )
        return target, equivalence_map

    def _direct(
        self,
        source: Dict[str, Tensor],
        key: str,
        pairs: List[EquivalencePair],
        cka_scores: Dict[str, float],
        swap_fraction: float,
    ) -> Tensor:
        w = source[key]
        out = self._apply_fraction(w.clone(), swap_fraction)
        pairs.append(self._pair(key, key, out.shape, w, out, SwapType.DIRECT))
        cka_scores[key] = 1.0 if swap_fraction >= 1.0 else weight_cka(w, out)
        return out

    def _norm_from_layernorm(
        self,
        source: Dict[str, Tensor],
        src_key: str,
        tgt_key: str,
        pairs: List[EquivalencePair],
        cka_scores: Dict[str, float],
        swap_fraction: float,
    ) -> Tensor:
        w = source[src_key]
        out = self._apply_fraction(w.clone(), swap_fraction)
        pairs.append(self._pair(src_key, tgt_key, out.shape, w, out, SwapType.DIRECT))
        cka_scores[tgt_key] = weight_cka(w, out)
        return out

    def _projected(
        self,
        source: Tensor,
        target_shape: tuple,
        swap_fraction: float,
    ) -> Tensor:
        projected = self.swap_engine.swap(
            source, target_shape, SwapType.PROJECTED, importance_mask=None
        )
        return self._apply_fraction(projected, swap_fraction)

    def _apply_fraction(self, tensor: Tensor, swap_fraction: float) -> Tensor:
        if swap_fraction >= 1.0:
            return tensor
        mask = self.scorer.score_svd_spectrum(tensor, top_k_fraction=swap_fraction)
        if mask.shape != tensor.shape:
            return tensor
        init = self.swap_engine._xavier_init(tensor.shape, tensor.dtype, tensor.device)
        init[mask] = tensor[mask]
        return init

    def _pair(
        self,
        src_name: str,
        tgt_name: str,
        tgt_shape: tuple,
        src_tensor: Tensor,
        tgt_tensor: Tensor,
        swap_type: SwapType,
        confidence: float = 0.85,
    ) -> EquivalencePair:
        cka = weight_cka(src_tensor, tgt_tensor) if src_tensor.shape == tgt_tensor.shape else weight_cka(
            src_tensor.reshape(src_tensor.shape[0], -1),
            tgt_tensor.reshape(tgt_tensor.shape[0], -1),
        )
        wp = WeightProduct(
            name=src_name,
            tensor=src_tensor,
            shape=tuple(src_tensor.shape),
            functional_role=FunctionalRole.UNKNOWN,
            paradigm="llm",
            architecture="tinytransformer",
            layer_index=-1,
        )
        return EquivalencePair(
            source=wp,
            target_layer_name=tgt_name,
            target_shape=tgt_shape,
            cka_score=cka,
            swap_type=swap_type,
            confidence=confidence,
        )

    def _unmapped(self, source: Dict[str, Tensor], pairs: List[EquivalencePair]) -> List[str]:
        used = set()
        for p in pairs:
            used.add(p.source.name.split(" (derived)")[0].split("+")[0])
            for k in source:
                if k.startswith(p.source.name.split("+")[0].split(" (derived)")[0]):
                    used.add(k)
        mapped_keys = {
            "embedding.weight", "lm_head.weight", "norm.weight",
        }
        for i in range(self.N_LAYERS):
            p = f"layers.{i}."
            mapped_keys.update({
                f"{p}q_proj.weight", f"{p}k_proj.weight", f"{p}v_proj.weight",
                f"{p}o_proj.weight", f"{p}gate_proj.weight", f"{p}up_proj.weight",
                f"{p}down_proj.weight", f"{p}input_layernorm.weight",
                f"{p}post_attention_layernorm.weight",
            })
        return [k for k in source if k not in mapped_keys]

    def _quality_tier(self, mean_cka: float) -> QualityTier:
        if mean_cka >= 0.85:
            return QualityTier.EXCELLENT
        if mean_cka >= 0.70:
            return QualityTier.GOOD
        if mean_cka >= 0.55:
            return QualityTier.ACCEPTABLE
        return QualityTier.DEGRADED
