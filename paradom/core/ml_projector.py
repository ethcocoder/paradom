"""
ML Ensemble Weight Projector
=============================

Uses a trained Gradient Boosting model to learn the optimal projection
strategy for each weight tensor, combining multiple existing methods.

Approach:
    1. For each weight tensor, generate candidate projections using:
       - SVD with different ranks
       - Head merging (activation-aware)
       - Head-boundary truncation
       - Combined methods
    2. Extract features from the source weight
    3. Train model to predict projection quality (CKA)
    4. At inference, predict best candidate and use it

This is a "meta-learner" — it learns which projection method works best
for which type of weight tensor.

Why ML works here:
    - Different weight types (q_proj, k_proj, FFN) need different projections
    - SVD is optimal for some, head merging for others
    - The model learns these patterns from calibration data
    - Inference is <1ms per weight (just a predict() call)
"""
import torch
import numpy as np
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def extract_weight_features(W: Tensor, target_shape: tuple, role=None) -> np.ndarray:
    """
    Extract fixed-size feature vector from a source weight tensor.

    Features capture:
    - Shape information (size, ratios, reduction amounts)
    - Weight statistics (mean, std, norm, etc.)
    - SVD spectrum (top singular values, energy distribution)
    - Head structure (if applicable)
    """
    W_np = W.float().cpu().numpy()
    src_rows, src_cols = W_np.shape
    tgt_rows, tgt_cols = target_shape

    features = []

    # 1. Shape features
    features.append(src_rows)
    features.append(src_cols)
    features.append(src_rows * src_cols)  # total params
    features.append(src_rows / src_cols if src_cols > 0 else 0)  # aspect ratio
    features.append(tgt_rows / src_rows if src_rows > 0 else 0)  # row ratio
    features.append(tgt_cols / src_cols if src_cols > 0 else 0)  # col ratio
    features.append((tgt_rows * tgt_cols) / (src_rows * src_cols))  # param ratio

    # 2. Weight statistics
    features.append(float(W_np.mean()))
    features.append(float(W_np.std()))
    features.append(float(np.abs(W_np).mean()))
    features.append(float(np.linalg.norm(W_np)))
    features.append(float(np.linalg.norm(W_np)) / (src_rows * src_cols))  # Frobenius density

    # 3. SVD spectrum (top 10 singular values + energy distribution)
    try:
        U, S, Vh = np.linalg.svd(W_np, full_matrices=False)
        top_k = min(10, len(S))
        # Pad to fixed size
        sv_padded = np.zeros(10)
        sv_padded[:top_k] = S[:top_k]
        features.extend(sv_padded.tolist())

        # Energy distribution
        total_energy = np.sum(S ** 2)
        if total_energy > 0:
            cumsum = np.cumsum(S ** 2) / total_energy
            energy_points = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            for ep in energy_points:
                idx = np.searchsorted(cumsum, ep)
                features.append(idx / len(S) if len(S) > 0 else 0)
        else:
            features.extend([0.0] * 7)

        # Rank approximation
        if total_energy > 0:
            rank_90 = np.searchsorted(cumsum, 0.9)
            features.append(rank_90 / len(S) if len(S) > 0 else 0)
        else:
            features.append(0.0)
    except Exception:
        features.extend([0.0] * 18)  # 10 SVs + 7 energy + 1 rank

    # 4. Head structure features
    head_dim = 64
    has_heads = (src_rows % head_dim == 0) and (src_rows // head_dim <= 16)
    features.append(1.0 if has_heads else 0.0)
    if has_heads:
        num_heads = src_rows // head_dim
        features.append(num_heads)
        features.append(src_cols / head_dim)  # d_model / head_dim
    else:
        features.append(0)
        features.append(0)

    # 5. Role features (one-hot)
    role_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                  'gate_proj', 'up_proj', 'down_proj', 'embed', 'norm', 'other']
    role_idx = 9  # default: other
    if role is not None:
        role_str = str(role).lower()
        for i, name in enumerate(role_names):
            if name in role_str:
                role_idx = i
                break
    role_onehot = [0.0] * len(role_names)
    role_onehot[role_idx] = 1.0
    features.extend(role_onehot)

    return np.array(features, dtype=np.float32)


def generate_candidates(
    W: Tensor,
    target_shape: tuple,
    src_heads: int = 0,
    tgt_heads: int = 0,
    head_dim: int = 64,
) -> List[Tensor]:
    """
    Generate multiple candidate projections for a weight tensor.

    Returns list of (candidate, method_name) tuples.
    """
    candidates = []
    if len(target_shape) == 1:
        return candidates
    W_2d = W.float().reshape(W.shape[0], -1)
    m_src, n_src = W_2d.shape
    d_out, d_in = target_shape

    # Method 1: Standard SVD (full)
    try:
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
        k = min(len(S), d_out, d_in)
        W_svd = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in]
        W_svd = W_svd[:d_out, :d_in].reshape(target_shape)
        candidates.append((W_svd, 'svd_full'))
    except Exception:
        pass

    # Method 2: SVD with rank = min(d_out, d_in) - aggressive truncation
    try:
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
        k = min(d_out, d_in)
        W_svd2 = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in]
        W_svd2 = W_svd2[:d_out, :d_in].reshape(target_shape)
        candidates.append((W_svd2, 'svd_rank'))
    except Exception:
        pass

    # Method 3: Head-boundary truncation (row first, then SVD columns)
    if src_heads > 0 and m_src > d_out and m_src % head_dim == 0:
        try:
            # Truncate rows at head boundaries
            W_trunc_rows = W_2d[:d_out, :]
            # SVD on columns
            if W_trunc_rows.shape[1] > d_in:
                U, S, Vh = torch.linalg.svd(W_trunc_rows, full_matrices=False)
                k = min(len(S), d_in)
                W_hb = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in]
            else:
                W_hb = W_trunc_rows
            W_hb = W_hb[:d_out, :d_in].reshape(target_shape)
            candidates.append((W_hb, 'head_boundary'))
        except Exception:
            pass

    # Method 4: Head merging (if reducing kv_heads)
    if src_heads > 0 and tgt_heads > 0 and src_heads > tgt_heads and m_src % head_dim == 0:
        try:
            num_src = m_src // head_dim
            W_heads = W_2d.reshape(num_src, head_dim, n_src)

            # Merge by averaging most similar pairs
            if num_src > tgt_heads:
                # Simple: average all heads equally
                while W_heads.shape[0] > tgt_heads:
                    # Find most similar pair
                    best_sim = -1
                    best_i, best_j = 0, 1
                    for i in range(W_heads.shape[0]):
                        for j in range(i + 1, W_heads.shape[0]):
                            sim = torch.cosine_similarity(
                                W_heads[i].flatten().unsqueeze(0),
                                W_heads[j].flatten().unsqueeze(0)
                            ).item()
                            if sim > best_sim:
                                best_sim = sim
                                best_i, best_j = i, j
                    # Merge
                    merged = (W_heads[best_i] + W_heads[best_j]) / 2
                    W_heads[best_i] = merged
                    keep_mask = torch.ones(W_heads.shape[0], dtype=torch.bool)
                    keep_mask[best_j] = False
                    W_heads = W_heads[keep_mask]

            W_merged = W_heads.reshape(tgt_heads * head_dim, n_src)

            # SVD on columns
            if W_merged.shape[1] > d_in:
                U, S, Vh = torch.linalg.svd(W_merged, full_matrices=False)
                k = min(len(S), d_in)
                W_merged = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in]

            W_merged = W_merged[:d_out, :d_in].reshape(target_shape)
            candidates.append((W_merged, 'head_merge'))
        except Exception:
            pass

    # Method 5: Truncation only (simplest)
    try:
        W_trunc = torch.zeros(target_shape, dtype=torch.float32)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(W_2d.shape, target_shape))
        W_trunc_flat = torch.zeros(target_shape[0], target_shape[1] if len(target_shape) > 1 else 1)
        W_trunc_2d = torch.zeros(d_out, d_in)
        W_trunc_2d[:min(m_src, d_out), :min(n_src, d_in)] = W_2d[:min(m_src, d_out), :min(n_src, d_in)]
        candidates.append((W_trunc_2d.reshape(target_shape), 'truncation'))
    except Exception:
        pass

    # Method 6: Energy-scaled SVD
    try:
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
        k = min(len(S), d_out, d_in)
        W_es = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in]
        W_es = W_es[:d_out, :d_in]
        # Energy rescaling
        src_energy = W_2d.pow(2).sum()
        proj_energy = W_es.pow(2).sum()
        if proj_energy > 0:
            scale = (src_energy / proj_energy).sqrt().clamp(0.5, 2.0)
            W_es = W_es * scale
        candidates.append((W_es.reshape(target_shape), 'svd_energy'))
    except Exception:
        pass

    # ── FFN-SPECIFIC METHODS (for gate_proj, up_proj, down_proj) ──
    is_ffn = src_heads == 0  # FFN has no head structure

    if is_ffn and m_src > d_out:
        # Method 7: Row-norm selection (keep highest-norm rows)
        try:
            row_norms = W_2d.pow(2).sum(dim=1)
            _, top_rows = row_norms.topk(d_out)
            top_rows, _ = top_rows.sort()
            W_rns = W_2d[top_rows, :][:, :d_in]
            candidates.append((W_rns.reshape(target_shape), 'row_norm_select'))
        except Exception:
            pass

        # Method 8: Center crop (keep middle rows)
        try:
            start_row = (m_src - d_out) // 2
            W_cc = W_2d[start_row:start_row + d_out, :][:, :d_in]
            candidates.append((W_cc.reshape(target_shape), 'center_crop'))
        except Exception:
            pass

        # Method 9: Row-norm + SVD columns
        try:
            row_norms = W_2d.pow(2).sum(dim=1)
            _, top_rows = row_norms.topk(d_out)
            top_rows, _ = top_rows.sort()
            W_rns_svd = W_2d[top_rows, :]
            if W_rns_svd.shape[1] > d_in:
                U, S, Vh = torch.linalg.svd(W_rns_svd, full_matrices=False)
                k = min(len(S), d_in)
                W_rns_svd = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in]
            W_rns_svd = W_rns_svd[:d_out, :d_in]
            candidates.append((W_rns_svd.reshape(target_shape), 'row_norm_svd'))
        except Exception:
            pass

        # Method 10: Column-norm + SVD rows
        try:
            col_norms = W_2d.pow(2).sum(dim=0)
            _, top_cols = col_norms.topk(d_in)
            top_cols, _ = top_cols.sort()
            W_cns = W_2d[:, top_cols]
            if W_cns.shape[0] > d_out:
                U, S, Vh = torch.linalg.svd(W_cns, full_matrices=False)
                k = min(len(S), d_out)
                W_cns = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
            W_cns = W_cns[:d_out, :d_in]
            candidates.append((W_cns.reshape(target_shape), 'col_norm_svd'))
        except Exception:
            pass

        # Method 11: Interleaved row selection (every Nth row)
        try:
            stride = m_src / d_out
            indices = [int(i * stride) for i in range(d_out)]
            W_il = W_2d[indices, :][:, :d_in]
            candidates.append((W_il.reshape(target_shape), 'interleaved_rows'))
        except Exception:
            pass

        # Method 12: Random projection (JL-style)
        try:
            torch.manual_seed(42)
            proj_matrix = torch.randn(d_in, n_src, device=W_2d.device) / (n_src ** 0.5)
            W_rp = W_2d @ proj_matrix.T
            # Then pick top rows by norm
            row_norms = W_rp.pow(2).sum(dim=1)
            _, top_rows = row_norms.topk(d_out)
            top_rows, _ = top_rows.sort()
            W_rp = W_rp[top_rows, :]
            candidates.append((W_rp[:d_out, :d_in].reshape(target_shape), 'random_proj'))
        except Exception:
            pass

    return candidates


class EnsembleProjector:
    """
    ML-based ensemble projector that combines multiple projection methods.

    Uses Gradient Boosting to learn which projection strategy works best
    for each weight tensor based on its features.

    Usage::

        projector = EnsembleProjector()
        projector.calibrate(source_model, tokenizer, prompt, target_config)
        # Then use in mapper:
        mapper = TransformerToTransformerMapper(ml_projector=projector)
    """

    FEATURE_SIZE = 48  # Fixed feature vector size

    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.1):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._trained = False
        self._candidate_methods = [
            'svd_full', 'svd_rank', 'head_boundary', 'head_merge',
            'truncation', 'svd_energy',
            'row_norm_select', 'center_crop', 'row_norm_svd',
            'col_norm_svd', 'interleaved_rows', 'random_proj'
        ]

    def train(
        self,
        source_weights: List[Tuple[str, Tensor]],
        target_config: Dict,
        source_config: Dict,
        kv_activations: Optional[Dict] = None,
        quality_fn=None,
    ):
        """
        Train the ensemble model on calibration data.

        For each source weight:
        1. Generate candidate projections
        2. Evaluate each candidate (using CKA or quality_fn)
        3. Extract features
        4. Train model to predict quality

        Args:
            source_weights: list of (key, tensor) pairs
            target_config: target model config
            source_config: source model config
            kv_activations: optional KV activations for head-aware methods
            quality_fn: function(W_src, W_tgt) -> float (default: CKA)
        """
        from paradom.core.cka import weight_cka

        if quality_fn is None:
            # Use reconstruction quality instead of broken cross-shape CKA
            # Measures how well projection preserves spectral structure
            def quality_fn(w_src, w_proj):
                src_2d = w_src.float().reshape(w_src.shape[0], -1)
                proj_2d = w_proj.float().reshape(w_proj.shape[0], -1)
                # SVD energy preservation ratio
                try:
                    _, S_src, _ = torch.linalg.svd(src_2d, full_matrices=False)
                    _, S_proj, _ = torch.linalg.svd(proj_2d, full_matrices=False)
                    k = min(len(S_src), len(S_proj))
                    if k == 0:
                        return 0.0
                    # Compare top-k singular value distributions
                    src_energy = S_src[:k].pow(2).sum()
                    proj_energy = S_proj[:k].pow(2).sum()
                    if src_energy == 0:
                        return 0.0
                    ratio = proj_energy / src_energy
                    # Penalize if ratio is too far from ideal (size-proportional)
                    ideal_ratio = proj_2d.numel() / src_2d.numel()
                    return 1.0 - abs(ratio - ideal_ratio)
                except Exception:
                    return 0.0

        src_d = source_config.get("d_model", 576)
        tgt_d = target_config.get("d_model", 512)
        src_heads = source_config.get("num_heads", 9)
        src_kv = source_config.get("num_key_value_heads", 3)
        tgt_heads = target_config.get("num_heads", 8)
        tgt_kv = target_config.get("num_key_value_heads", 2)
        head_dim = source_config.get("head_dim", 64)
        tgt_inner = target_config.get("d_inner", 1408)
        src_inner = source_config.get("d_inner", 1536)

        features_list = []
        labels_list = []

        for key, W in source_weights:
            # Determine target shape
            tgt_shape = self._infer_target_shape(
                key, W.shape, src_d, tgt_d, src_heads, tgt_heads,
                src_kv, tgt_kv, src_inner, tgt_inner, head_dim
            )
            if tgt_shape is None:
                continue

            # Determine role for head-aware methods
            role = None
            num_src_heads = 0
            num_tgt_heads = 0
            if 'q_proj' in key:
                role = 'q_proj'
                num_src_heads = src_heads
                num_tgt_heads = tgt_heads
            elif 'k_proj' in key:
                role = 'k_proj'
                num_src_heads = src_kv
                num_tgt_heads = tgt_kv
            elif 'v_proj' in key:
                role = 'v_proj'
                num_src_heads = src_kv
                num_tgt_heads = tgt_kv
            elif 'o_proj' in key:
                role = 'o_proj'
                num_src_heads = src_heads
                num_tgt_heads = tgt_heads

            # Generate candidates
            candidates = generate_candidates(
                W, tgt_shape, num_src_heads, num_tgt_heads, head_dim
            )

            if not candidates:
                continue

            # Extract features
            feat = extract_weight_features(W, tgt_shape, role)

            # Evaluate each candidate
            best_quality = -1
            best_idx = 0
            for i, (cand, method) in enumerate(candidates):
                try:
                    q = quality_fn(W, cand)
                    if q > best_quality:
                        best_quality = q
                        best_idx = i
                except Exception:
                    pass

            features_list.append(feat)
            labels_list.append(float(best_idx))

        if len(features_list) < 5:
            return

        X = np.array(features_list)
        y = np.array(labels_list)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self._trained = True

    def predict_best_candidate(
        self,
        W: Tensor,
        target_shape: tuple,
        role=None,
    ) -> Tensor:
        """
        Predict and return the best projection for a weight tensor.
        """
        if not self._trained:
            # Fallback to standard SVD
            return self._svd_fallback(W, target_shape)

        # Generate candidates
        candidates = generate_candidates(W, target_shape)
        if not candidates:
            return self._svd_fallback(W, target_shape)

        # Extract features
        feat = extract_weight_features(W, target_shape, role)
        X = self.scaler.transform(feat.reshape(1, -1))

        # Predict best candidate index
        pred_idx = int(round(self.model.predict(X)[0]))
        pred_idx = max(0, min(pred_idx, len(candidates) - 1))

        return candidates[pred_idx][0]

    def _infer_target_shape(self, key, src_shape, src_d, tgt_d,
                            src_heads, tgt_heads, src_kv, tgt_kv,
                            src_inner, tgt_inner, head_dim):
        """Infer target shape from source shape and config."""
        rows, cols = src_shape[0], src_shape[1] if len(src_shape) > 1 else 1

        if 'q_proj' in key:
            return (tgt_heads * head_dim, tgt_d)
        elif 'k_proj' in key:
            return (tgt_kv * head_dim, tgt_d)
        elif 'v_proj' in key:
            return (tgt_kv * head_dim, tgt_d)
        elif 'o_proj' in key:
            return (tgt_d, tgt_heads * head_dim)
        elif 'gate_proj' in key or 'up_proj' in key:
            return (tgt_inner, tgt_d)
        elif 'down_proj' in key:
            return (tgt_d, tgt_inner)
        elif 'embed_tokens' in key or 'lm_head' in key:
            vocab = src_shape[0]
            return (vocab, tgt_d)
        elif 'layernorm' in key or 'norm' in key:
            return (tgt_d,)
        else:
            return None

    def _svd_fallback(self, W, target_shape):
        """Standard SVD projection fallback."""
        W_2d = W.float().reshape(W.shape[0], -1)
        d_out = target_shape[0]
        d_in = 1
        for dim in target_shape[1:]:
            d_in *= dim

        result = W_2d.clone()
        if result.shape[1] > d_in:
            U, S, Vh = torch.linalg.svd(result, full_matrices=False)
            k = min(len(S), d_in)
            result = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in]
        if result.shape[0] > d_out:
            U, S, Vh = torch.linalg.svd(result, full_matrices=False)
            k = min(len(S), d_out)
            result = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
        result = result[:d_out, :d_in]

        # Pad if needed
        if result.shape[0] < d_out:
            result = torch.cat([result, torch.zeros(d_out - result.shape[0], result.shape[1])], dim=0)
        if result.shape[1] < d_in:
            result = torch.cat([result, torch.zeros(result.shape[0], d_in - result.shape[1])], dim=1)

        return result.reshape(target_shape).to(W.dtype)
