"""
Activation-Aware Weight Projector

Projects transformer weights while preserving attention patterns (functional
quality) rather than Frobenius norm (structural similarity).

Core insight: SVD gives CKA=0.9995 but garbage output because it mixes
information across head boundaries. The target expects clean 64-dim head
blocks, but SVD creates linear combinations of all source heads.

Algorithm:
    1. Run source model on calibration prompt, collect Q/K/V activations
    2. For each layer, compute per-head attention contribution and similarity
    3. Greedy merge the most similar head pair, weighted by importance
    4. Row projection (head merging) then column SVD (d_model reduction)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any


def collect_qkv_activations(
    source_model,
    tokenizer,
    prompt: str,
    max_length: int = 128,
) -> Dict[int, Dict[str, Tensor]]:
    """
    Run source model on a prompt and collect per-layer Q/K/V projection outputs.

    Returns:
        {layer_idx: {"Q": Tensor, "K": Tensor, "V": Tensor}}
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    activations: Dict[int, Dict[str, Tensor]] = {}
    hooks = []

    def make_hook(idx: int, key: str):
        def hook(module, inp, output):
            act = output[0].detach().float() if isinstance(output, tuple) else output.detach().float()
            activations.setdefault(idx, {})[key] = act
        return hook

    for idx, layer in enumerate(source_model.model.layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_hook(idx, "Q")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_hook(idx, "K")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_hook(idx, "V")))

    with torch.no_grad():
        source_model(**inputs)

    for h in hooks:
        h.remove()

    return activations


class ActivationAwareProjector:
    """
    Projects weight matrices while preserving attention patterns.

    Uses calibration data to identify important/redundant heads and merges
    them weighted by their attention contribution.

    Usage::

        projector = ActivationAwareProjector(source_config, target_config)
        projector.calibrate(source_model, tokenizer, "Hello, world")
        W_tgt = projector.project(W_src, (128, 512), layer_idx=0, role=FunctionalRole.CONTEXT_KEY)
    """

    def __init__(
        self,
        source_config: Dict[str, Any],
        target_config: Dict[str, Any],
        head_dim: int = 64,
        calibration_data: Optional[Dict[int, Dict[str, Tensor]]] = None,
    ):
        self.src_config = source_config
        self.tgt_config = target_config
        self.head_dim = head_dim

        self.src_num_heads = source_config.get("num_heads", 9)
        self.src_num_kv = source_config.get("num_key_value_heads", 3)
        self.tgt_num_heads = target_config.get("num_heads", 8)
        self.tgt_num_kv = target_config.get("num_key_value_heads", 2)

        self._calibration: Dict[int, Dict[str, Tensor]] = {}
        self._head_scores: Dict[int, Dict[str, Tensor]] = {}

        if calibration_data is not None:
            self._calibration = calibration_data
            self._precompute_head_scores()

    def calibrate(self, source_model, tokenizer, prompt: str, max_length: int = 128):
        """Run calibration on a source model with a short prompt."""
        self._calibration = collect_qkv_activations(source_model, tokenizer, prompt, max_length)
        self._precompute_head_scores()

    def _precompute_head_scores(self):
        """Pre-compute head importance and similarity for all calibrated layers."""
        for idx, acts in self._calibration.items():
            if "Q" in acts and "K" in acts and "V" in acts:
                self._head_scores[idx] = self._compute_head_scores(
                    acts["Q"], acts["K"], acts["V"]
                )

    # ------------------------------------------------------------------
    # Head scoring
    # ------------------------------------------------------------------

    def _compute_head_scores(self, Q: Tensor, K: Tensor, V: Tensor) -> Dict[str, Tensor]:
        """
        Compute attention-based importance and similarity for all heads.

        importance[h] = how much head h contributes to attention output
        similarity[h1,h2] = how similar are their attention patterns
        """
        T = Q.shape[0]
        d = self.head_dim
        num_q = self.src_num_heads
        num_kv = self.src_num_kv

        Q_heads = Q.reshape(T, num_q, d)
        K_heads = K.reshape(T, num_kv, d)
        V_heads = V.reshape(T, num_kv, d)

        attn_weighted_v = torch.zeros(num_q, num_kv, d)
        attn_flat = torch.zeros(num_q, num_kv, T * T)

        scale = d ** 0.5
        for i in range(num_q):
            for h in range(num_kv):
                scores = Q_heads[:, i, :] @ K_heads[:, h, :].T / scale
                attn = F.softmax(scores, dim=-1)
                attn_flat[i, h] = attn.reshape(-1)
                attn_weighted_v[i, h] = (attn @ V_heads[:, h, :]).mean(dim=0)

        kv_importance = attn_weighted_v.abs().sum(dim=(0, 2))
        q_importance = attn_flat.abs().sum(dim=(1, 2, 3))

        kv_similarity = torch.zeros(num_kv, num_kv)
        for h1 in range(num_kv):
            for h2 in range(num_kv):
                sims = [
                    F.cosine_similarity(attn_weighted_v[i, h1].unsqueeze(0),
                                        attn_weighted_v[i, h2].unsqueeze(0))
                    for i in range(num_q)
                ]
                kv_similarity[h1, h2] = torch.stack(sims).mean()

        q_similarity = torch.zeros(num_q, num_q)
        for i1 in range(num_q):
            for i2 in range(num_q):
                sims = [
                    F.cosine_similarity(attn_flat[i1, h].unsqueeze(0),
                                        attn_flat[i2, h].unsqueeze(0))
                    for h in range(num_kv)
                ]
                q_similarity[i1, i2] = torch.stack(sims).mean()

        return {
            "kv_importance": kv_importance,
            "q_importance": q_importance,
            "kv_similarity": kv_similarity,
            "q_similarity": q_similarity,
        }

    # ------------------------------------------------------------------
    # Main projection
    # ------------------------------------------------------------------

    def project(self, W_src: Tensor, target_shape: tuple,
                layer_idx: int, role) -> Tensor:
        """
        Project a weight matrix to target shape using activation-aware method.

        Routes by FunctionalRole to handle head dimension correctly.
        Falls back to SVD for non-attention weights.
        """
        from .enums import FunctionalRole

        if layer_idx not in self._head_scores:
            return self._svd_fallback(W_src, target_shape)

        scores = self._head_scores[layer_idx]
        W_2d = W_src.float().reshape(W_src.shape[0], -1)

        if role in (FunctionalRole.CONTEXT_KEY, FunctionalRole.CONTEXT_VALUE):
            return self._project_with_head_merging(
                W_2d, target_shape,
                scores["kv_importance"], scores["kv_similarity"],
                self.src_num_kv, self.tgt_num_kv, is_row=True,
            )
        elif role == FunctionalRole.CONTEXT_QUERY:
            return self._project_with_head_merging(
                W_2d, target_shape,
                scores["q_importance"], scores["q_similarity"],
                self.src_num_heads, self.tgt_num_heads, is_row=True,
            )
        elif role == FunctionalRole.CONTEXT_OUTPUT:
            return self._project_with_head_merging(
                W_2d, target_shape,
                scores["q_importance"], scores["q_similarity"],
                self.src_num_heads, self.tgt_num_heads, is_row=False,
            )
        else:
            return self._svd_fallback(W_src, target_shape)

    # ------------------------------------------------------------------
    # Head-aware projection
    # ------------------------------------------------------------------

    def _project_with_head_merging(
        self, W_2d, target_shape, head_importance, head_similarity,
        src_heads, tgt_heads, is_row,
    ):
        """Head merging for head dimension, SVD for d_model."""
        d_out_tgt = target_shape[0]
        d_in_tgt = 1
        for dim in target_shape[1:]:
            d_in_tgt *= dim

        if is_row:
            if src_heads > tgt_heads:
                W_merged = self._merge_heads(
                    W_2d, head_importance, head_similarity,
                    src_heads, tgt_heads, self.head_dim,
                )
            else:
                W_merged = W_2d

            if W_merged.shape[1] > d_in_tgt:
                U, S, Vh = torch.linalg.svd(W_merged, full_matrices=False)
                k = min(len(S), d_in_tgt)
                W_final = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in_tgt]
            else:
                W_final = W_merged
        else:
            if src_heads > tgt_heads:
                W_T = W_2d.T.contiguous()
                W_merged = self._merge_heads(
                    W_T, head_importance, head_similarity,
                    src_heads, tgt_heads, self.head_dim,
                )
                W_final = W_merged.T.contiguous()
            else:
                W_final = W_2d

            if W_final.shape[0] > d_out_tgt:
                U, S, Vh = torch.linalg.svd(W_final, full_matrices=False)
                k = min(len(S), d_out_tgt)
                W_final = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]

        if W_final.shape[0] < d_out_tgt:
            pad = d_out_tgt - W_final.shape[0]
            W_final = torch.cat([W_final, torch.zeros(pad, W_final.shape[1])], dim=0)
        if W_final.shape[1] < d_in_tgt:
            pad = d_in_tgt - W_final.shape[1]
            W_final = torch.cat([W_final, torch.zeros(W_final.shape[0], pad)], dim=1)

        src_energy = W_2d.pow(2).sum()
        proj_energy = W_final[:d_out_tgt, :d_in_tgt].pow(2).sum()
        if proj_energy > 0:
            scale = (src_energy / proj_energy).sqrt().clamp(1.0, 2.0)
            W_final = W_final * scale

        return W_final[:d_out_tgt, :d_in_tgt].reshape(target_shape).to(W_src.dtype)

    # ------------------------------------------------------------------
    # Greedy head merging
    # ------------------------------------------------------------------

    def _merge_heads(self, W_2d, head_importance, head_similarity,
                     src_heads, tgt_heads, head_dim):
        """
        Merge weight rows from src_heads to tgt_heads via greedy similar-pair
        merging weighted by attention importance.

        For 3->2 reduction there are only 3 possible pairs, so greedy = optimal.
        """
        d_in = W_2d.shape[1]
        W_heads = W_2d.reshape(src_heads, head_dim, d_in).clone()
        active_importance = head_importance.clone()
        active_indices = list(range(src_heads))

        while len(active_indices) > tgt_heads:
            n = len(active_indices)
            best_sim = -float("inf")
            best_pair = (0, 1)
            for i in range(n):
                for j in range(i + 1, n):
                    sim = head_similarity[active_indices[i], active_indices[j]].item()
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)

            i, j = best_pair
            w_i = active_importance[i] / (active_importance[i] + active_importance[j] + 1e-8)
            merged = w_i * W_heads[i] + (1.0 - w_i) * W_heads[j]

            W_heads[i] = merged
            active_importance[i] = active_importance[i] + active_importance[j]
            active_indices.pop(j)

            keep_mask = torch.ones(W_heads.shape[0], dtype=torch.bool)
            keep_mask[j] = False
            W_heads = W_heads[keep_mask]
            active_importance = active_importance[keep_mask]

        return W_heads.reshape(tgt_heads * head_dim, d_in)

    # ------------------------------------------------------------------
    # SVD fallback
    # ------------------------------------------------------------------

    def _svd_fallback(self, W_src, target_shape):
        """Standard SVD projection for non-attention weights."""
        W_2d = W_src.float().reshape(W_src.shape[0], -1)
        d_out_tgt = target_shape[0]
        d_in_tgt = 1
        for dim in target_shape[1:]:
            d_in_tgt *= dim

        src_energy = W_2d.pow(2).sum()
        result = W_2d.clone()

        if result.shape[1] > d_in_tgt:
            U, S, Vh = torch.linalg.svd(result, full_matrices=False)
            k = min(len(S), d_in_tgt)
            result = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in_tgt]

        if result.shape[0] > d_out_tgt:
            U, S, Vh = torch.linalg.svd(result, full_matrices=False)
            k = min(len(S), d_out_tgt)
            result = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]

        if result.shape[0] < d_out_tgt:
            pad = d_out_tgt - result.shape[0]
            result = torch.cat([result, torch.zeros(pad, result.shape[1])], dim=0)
        if result.shape[1] < d_in_tgt:
            pad = d_in_tgt - result.shape[1]
            result = torch.cat([result, torch.zeros(result.shape[0], pad)], dim=1)

        proj_energy = result[:d_out_tgt, :d_in_tgt].pow(2).sum()
        if proj_energy > 0:
            scale = (src_energy / proj_energy).sqrt().clamp(1.0, 2.0)
            result = result * scale

        return result[:d_out_tgt, :d_in_tgt].reshape(target_shape).to(W_src.dtype)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_head_report(self, layer_idx: int) -> Optional[Dict[str, Any]]:
        """Return head importance/similarity report for debugging."""
        if layer_idx not in self._head_scores:
            return None
        scores = self._head_scores[layer_idx]
        kv_sim = scores["kv_similarity"]
        q_sim = scores["q_similarity"]

        best_kv_pair, best_kv_sim = (0, 1), -1.0
        for i in range(self.src_num_kv):
            for j in range(i + 1, self.src_num_kv):
                if kv_sim[i, j] > best_kv_sim:
                    best_kv_sim = kv_sim[i, j].item()
                    best_kv_pair = (i, j)

        best_q_pair, best_q_sim = (0, 1), -1.0
        for i in range(self.src_num_heads):
            for j in range(i + 1, self.src_num_heads):
                if q_sim[i, j] > best_q_sim:
                    best_q_sim = q_sim[i, j].item()
                    best_q_pair = (i, j)

        return {
            "layer": layer_idx,
            "kv_importance": scores["kv_importance"].tolist(),
            "kv_best_merge": best_kv_pair,
            "kv_merge_similarity": best_kv_sim,
            "q_importance": scores["q_importance"].tolist(),
            "q_best_merge": best_q_pair,
            "q_merge_similarity": best_q_sim,
        }
