import torch
import math
from torch import Tensor
from typing import Optional, Dict, List, Tuple
from .enums import SwapType, FunctionalRole


def collect_kv_activations(source_model, tokenizer, prompt, max_length=128):
    """
    Run source model on a prompt and collect per-layer k/v projection outputs.
    
    Returns:
        dict: {layer_idx: {"k": Tensor, "v": Tensor}} where each tensor
              is (seq_len, num_kv_heads * head_dim).
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    device = next(source_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    activations = {}
    hooks = []

    def make_k_hook(idx):
        def hook(module, inp, output):
            activations.setdefault(idx, {})["k"] = output[0].detach().float() if isinstance(output, tuple) else output.detach().float()
        return hook

    def make_v_hook(idx):
        def hook(module, inp, output):
            activations.setdefault(idx, {})["v"] = output[0].detach().float() if isinstance(output, tuple) else output.detach().float()
        return hook

    for idx, layer in enumerate(source_model.model.layers):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(idx)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(idx)))

    with torch.no_grad():
        source_model(**inputs)

    for h in hooks:
        h.remove()

    return activations


class MagneticProjector:
    """
    Population-level 'magnetic field' shared projector (LAZY).

    Instead of computing a spectral basis from a single weight matrix, this
    computes the shared spectral structure across ALL layers of the same
    functional role — the 'magnetic field' that attracts same-type agents.

    Uses mean + QR for Grassmannian averaging (works for ALL matrix shapes):
    1. Collect all weight matrices W_i of a given functional role
    2. Compute SVD of each: W_i = U_i @ S_i @ Vh_i
    3. Align all Vh_i to reference via reference-based sign resolution
    4. Compute mean Vh across all layers
    5. QR-orthogonalize the mean to form the shared orthonormal basis

    NOTE: Gram eigendecomposition (G = sum Vh^T Vh) fails when Vh is square
    (orthonormal rows → G = I → degenerate eigendecomposition). Mean + QR
    works for both square and non-square Vh matrices.

    RIGHT bases are computed from W (shared right singular vectors).
    LEFT bases are computed from W^T (shared left singular vectors).
    """

    def __init__(self):
        self._shared_bases: Dict[str, Tensor] = {}
        self._pending: Dict[str, List[Tensor]] = {}
        self._pending_transpose: Dict[str, bool] = {}

    def register_role(self, role_key: str, weight_matrices: List[Tensor], transpose: bool = False):
        if role_key not in self._shared_bases:
            self._pending[role_key] = weight_matrices
            self._pending_transpose[role_key] = transpose

    def get_basis(self, role_key: str) -> Optional[Tensor]:
        if role_key in self._shared_bases:
            return self._shared_bases[role_key]
        if role_key not in self._pending:
            return None

        weight_matrices = self._pending.pop(role_key)
        transpose = self._pending_transpose.pop(role_key)

        if len(weight_matrices) == 0:
            return None

        shared_basis = self._compute_mean_basis(weight_matrices, transpose)
        self._shared_bases[role_key] = shared_basis
        return shared_basis

    def _prepare(self, W: Tensor, transpose: bool) -> Tensor:
        W_2d = W.float().reshape(W.shape[0], -1)
        return W_2d.T if transpose else W_2d

    def _compute_mean_basis(self, weight_matrices: List[Tensor], transpose: bool) -> Tensor:
        """Compute shared basis via mean + QR (Grassmannian Karcher mean approximation)."""
        ref_W = self._prepare(weight_matrices[0], transpose)
        ref_svd = torch.linalg.svd(ref_W, full_matrices=False)
        ref_Vh = ref_svd.Vh.clone()
        n_comp = ref_Vh.shape[0]

        # Reference-based sign alignment for all matrices
        aligned_Vhs = [ref_Vh[:n_comp]]
        for W in weight_matrices[1:]:
            W_2d = self._prepare(W, transpose)
            _, _, Vh_i = torch.linalg.svd(W_2d, full_matrices=False)
            Vh_i_aligned = self._sign_align(Vh_i[:n_comp], ref_Vh[:n_comp])
            aligned_Vhs.append(Vh_i_aligned)

        # Mean + QR (works for both square and non-square Vh)
        mean_Vh = torch.stack(aligned_Vhs).mean(dim=0)
        Q, _ = torch.linalg.qr(mean_Vh.to(torch.float32).T)
        shared_basis = Q.T
        return shared_basis

    def _sign_align(self, Vh_source: Tensor, Vh_target: Tensor) -> Tensor:
        """Reference-based sign alignment using full-vector dot products."""
        aligned = Vh_source.clone()
        for i in range(aligned.shape[0]):
            if (aligned[i] * Vh_target[i]).sum() < 0:
                aligned[i] = -aligned[i]
        return aligned


class SwapEngine:
    """
    Executes the actual weight swap operations.
    Uses MagneticProjector for population-aware downscaling.

    Projection flow for _ot_swap:
      W ∈ R^(m×n) → R^(m'×n'):
        - If n > n': W' = W @ P         (shared right projector, column reduction)
        - If m > m': W'' = Q^T @ W'     (shared left projector, row reduction)
        - Final: W''' = W'' * scale     (Frobenius-norm energy preservation)
    """

    def __init__(self, magnetic_projector: Optional[MagneticProjector] = None):
        self.magnetic_projector = magnetic_projector

    def swap(
        self,
        source_weight: Tensor,
        target_shape: tuple,
        swap_type: SwapType,
        importance_mask: Optional[Tensor] = None,
        axis_labels: Optional[tuple] = None,
        functional_role: Optional[FunctionalRole] = None,
        head_structure: Optional[Tuple[int, int, bool]] = None,
    ) -> Tensor:
        if swap_type == SwapType.DIRECT:
            return self._direct_swap(source_weight, target_shape, importance_mask)
        elif swap_type == SwapType.PROJECTED:
            return self._projected_swap(source_weight, target_shape, importance_mask, head_structure=head_structure)
        elif swap_type == SwapType.OT:
            return self._ot_swap(source_weight, target_shape, importance_mask, axis_labels, functional_role)
        elif swap_type == SwapType.SKIP:
            return self._xavier_init(target_shape, source_weight.dtype)
        else:
            raise NotImplementedError(f"Swap type {swap_type} not yet implemented for Phase 1.")

    def _direct_swap(self, W_src: Tensor, target_shape: tuple, mask: Optional[Tensor]) -> Tensor:
        if W_src.shape != target_shape:
            raise ValueError(f"Direct swap requires identical shapes: {W_src.shape} vs {target_shape}")
        if mask is None:
            return W_src.clone().detach()
        W_target = self._xavier_init(target_shape, W_src.dtype, W_src.device)
        mask = mask.to(W_src.device)
        W_target[mask] = W_src[mask]
        return W_target

    def _projected_swap(self, W_src: Tensor, target_shape: tuple, mask: Optional[Tensor], head_structure: Optional[Tuple[int, int, bool]] = None) -> Tensor:
        """Project source weight to target shape via head-aware truncation.

        When head_structure is provided, uses DETERMINISTIC truncation
        (row-first for Q/K/V, column-first for O) instead of SVD.
        This avoids GPU-dependent SVD non-determinism and preserves
        head boundary structure. SVD is only used for non-head weights.
        """
        if mask is not None and mask.shape == W_src.shape:
            W_src = W_src.clone()
            W_src[~mask] = 0.0

        W_2d = W_src.reshape(W_src.shape[0], -1).float()
        m_src, n_src = W_2d.shape
        d_out_tgt = target_shape[0]
        d_in_tgt = 1
        for dim in target_shape[1:]:
            d_in_tgt *= dim

        m_diff = m_src - d_out_tgt
        n_diff = n_src - d_in_tgt

        if m_diff <= 0 and n_diff <= 0:
            W_target_2d = torch.zeros((d_out_tgt, d_in_tgt), dtype=torch.float32, device=W_src.device)
            W_target_2d[:m_src, :n_src] = W_2d
            return W_target_2d.reshape(target_shape).to(W_src.dtype)

        result = W_2d.clone()

        has_heads = head_structure is not None
        transpose = has_heads and len(head_structure) > 2 and head_structure[2]

        if has_heads:
            if transpose:
                if n_diff > 0:
                    result = result[:, :d_in_tgt]
                if m_diff > 0:
                    result = result[:d_out_tgt, :]
            else:
                if m_diff > 0:
                    result = result[:d_out_tgt, :]
                if n_diff > 0:
                    result = result[:, :d_in_tgt]
        else:
            if n_diff > 0:
                U, S, Vh = torch.linalg.svd(result, full_matrices=False)
                k = min(len(S), d_in_tgt)
                result = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :d_in_tgt]
            if m_diff > 0:
                U, S, Vh = torch.linalg.svd(result, full_matrices=False)
                k = min(len(S), d_out_tgt)
                result = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
                result = result[:d_out_tgt, :]

        if result.shape[0] < d_out_tgt:
            pad = d_out_tgt - result.shape[0]
            result = torch.cat([result, torch.zeros(pad, result.shape[1])], dim=0)
        if result.shape[1] < d_in_tgt:
            pad = d_in_tgt - result.shape[1]
            result = torch.cat([result, torch.zeros(result.shape[0], pad)], dim=1)

        src_energy = W_2d.pow(2).sum()
        proj_energy = result[:d_out_tgt, :d_in_tgt].pow(2).sum()
        if proj_energy > 0:
            scale = (src_energy / proj_energy).sqrt().clamp(1.0, 2.0)
            result = result * scale

        W_target = result[:d_out_tgt, :d_in_tgt].reshape(target_shape).to(W_src.dtype)

        if mask is not None and mask.shape == target_shape:
            W_final = self._xavier_init(target_shape, W_src.dtype, W_src.device)
            W_final[mask] = W_target[mask]
            return W_final
        return W_target

    def _projected_swap_head_aware(
        self,
        W_src: Tensor,
        target_shape: tuple,
        mask: Optional[Tensor],
        head_structure: Tuple[int, int, bool],
    ) -> Tensor:
        """
        Head-aware projection: SIMPLIFIES THE MATH, doesn't destroy it.

        When reducing heads (9→8), merges the two most SIMILAR heads
        (highest cosine similarity) — combining their information rather
        than dropping one entirely. This is "replacing hard maths with
        simpler maths": same knowledge, fewer parallel computations.

        For d_model reduction: gradual step-by-step SVD (each step is
        nearly lossless).
        """
        src_num_heads, src_head_dim = head_structure[:2]
        transpose_for_heads = len(head_structure) > 2 and head_structure[2]
        W_2d = W_src.reshape(W_src.shape[0], -1).float()
        src_out, src_in = W_2d.shape

        if transpose_for_heads:
            W_2d = W_2d.T
            src_out, src_in = src_in, src_out

        tgt_num_heads = target_shape[0] // src_head_dim if not transpose_for_heads else target_shape[1] // src_head_dim
        tgt_d_model = target_shape[1] if len(target_shape) > 1 and not transpose_for_heads else target_shape[0]

        is_downscale = src_num_heads > tgt_num_heads or src_in > tgt_d_model

        if is_downscale:
            W_reshaped = W_2d.reshape(src_num_heads, src_head_dim, src_in)

            W_current = W_reshaped.clone()

            while W_current.shape[0] > tgt_num_heads:
                h = W_current.shape[0]
                flat = W_current.reshape(h, -1)
                sims = torch.nn.functional.cosine_similarity(flat.unsqueeze(0), flat.unsqueeze(1), dim=2)
                for i in range(h):
                    sims[i, i] = -1.0
                best = (sims + sims.T) / 2.0
                idx = best.reshape(-1).argmax()
                i, j = divmod(idx.item(), h)
                if i > j:
                    i, j = j, i

                merged = (W_current[i] + W_current[j]) / 2.0
                keep = [k for k in range(h) if k != j]
                keep[i] = -1
                remaining = W_current[[k for k in range(h) if k != j and k != i]]
                W_current = torch.cat([merged.unsqueeze(0), remaining], dim=0)

            if src_in > tgt_d_model:
                projected_heads = []
                for h_idx in range(W_current.shape[0]):
                    U, S, Vh = torch.linalg.svd(W_current[h_idx], full_matrices=False)
                    k = min(len(S), tgt_d_model)
                    W_h = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :tgt_d_model]
                    projected_heads.append(W_h)
                result = torch.stack(projected_heads).reshape(tgt_num_heads * src_head_dim, tgt_d_model)
            else:
                result = W_current.reshape(tgt_num_heads * src_head_dim, src_in)

        else:
            W_reshaped = W_2d.reshape(src_num_heads, src_head_dim, src_in)
            if src_in < tgt_d_model:
                projected_heads = []
                for h in range(src_num_heads):
                    U, S, Vh = torch.linalg.svd(W_reshaped[h], full_matrices=True)
                    if len(S) >= tgt_d_model:
                        W_h = (U[:, :tgt_d_model] * S[:tgt_d_model].unsqueeze(0)) @ Vh[:tgt_d_model, :tgt_d_model]
                    else:
                        W_h = torch.zeros(src_head_dim, tgt_d_model)
                        W_h[:, :len(S)] = (U[:, :len(S)] * S.unsqueeze(0)) @ Vh[:len(S), :tgt_d_model]
                    projected_heads.append(W_h)
                result = torch.stack(projected_heads).reshape(src_num_heads * src_head_dim, tgt_d_model)
            else:
                result = W_reshaped.reshape(src_out, src_in)

            if src_num_heads < tgt_num_heads:
                new_heads = self._xavier_init((tgt_num_heads - src_num_heads, src_head_dim, tgt_d_model), W_src.dtype)
                extra = new_heads.reshape(-1, tgt_d_model).float()
                result = torch.cat([result, extra], dim=0)

        src_energy = W_2d.pow(2).sum()
        proj_energy = result.pow(2).sum()
        if proj_energy > 0:
            scale = (src_energy / proj_energy).sqrt().clamp(1.0, 2.0)
            result = result * scale

        if transpose_for_heads:
            result = result.T

        W_target = torch.zeros(target_shape, dtype=torch.float32, device=W_src.device)
        out_rows = min(result.shape[0], target_shape[0])
        out_cols = min(result.shape[1], target_shape[1]) if len(target_shape) > 1 else out_rows
        W_target[:out_rows, :out_cols] = result[:out_rows, :out_cols]

        if mask is not None and mask.shape == target_shape:
            W_final = self._xavier_init(target_shape, W_src.dtype, W_src.device)
            W_final[mask] = W_target[mask]
            return W_final
        return W_target

    def _get_spectral_projector(self, src_dim: int, tgt_dim: int, direction: str, W_2d: Tensor, functional_role: Optional[FunctionalRole] = None) -> Tensor:
        """
        Get a spectral projection matrix (no caching — magnetic projector handles its own).

        Priority:
          1. Magnetic projector's shared basis (keyed by FunctionalRole + direction)
          2. Per-weight SVD fallback

        Fallback uses full_matrices=True only when tgt_dim exceeds SVD rank.
        """
        if self.magnetic_projector is not None and functional_role is not None:
            suffix = "_left" if direction == 'left' else "_right"
            role_key = f"role_{functional_role.name}{suffix}"
            Vh_shared = self.magnetic_projector.get_basis(role_key)
            if Vh_shared is not None and tgt_dim <= Vh_shared.shape[0]:
                return Vh_shared[:tgt_dim, :].T

        rank = min(W_2d.shape)
        need_full = tgt_dim > rank
        if direction == 'right':
            _, _, Vh = torch.linalg.svd(W_2d.float(), full_matrices=need_full)
            return Vh[:tgt_dim, :].T
        else:
            U, _, _ = torch.linalg.svd(W_2d.float(), full_matrices=need_full)
            return U[:, :tgt_dim]

    def _ot_swap(self, W_src: Tensor, target_shape: tuple, mask: Optional[Tensor], axis_labels: Optional[tuple] = None, functional_role: Optional[FunctionalRole] = None) -> Tensor:
        """
        Shared Spectral Projection for downscaling with magnetic compression.

        Unlike the original version, this:
        - Projects columns and rows via shared magnetic bases
        - Applies a SINGLE Frobenius-norm rescaling at the end (not per-projection SVD corrections)
        - Does NOT recompute SVD of intermediate results (saves ~500 SVDs per test)
        """
        d_out_tgt = target_shape[0]
        d_in_tgt = 1
        for dim in target_shape[1:]:
            d_in_tgt *= dim
        W_src_d_in = W_src.numel() // W_src.shape[0]

        if W_src.shape[0] < d_out_tgt or W_src_d_in < d_in_tgt:
            return self._projected_swap(W_src, target_shape, mask)

        W_2d = W_src.reshape(W_src.shape[0], -1).float()
        result = W_2d
        src_energy = W_2d.pow(2).sum()

        if W_src_d_in > d_in_tgt:
            P = self._get_spectral_projector(W_src_d_in, d_in_tgt, 'right', W_2d, functional_role)
            result = result @ P

        if W_src.shape[0] > d_out_tgt:
            Q = self._get_spectral_projector(W_src.shape[0], d_out_tgt, 'left', W_2d, functional_role)
            result = Q.T @ result

        # Magnetic compression: Frobenius-norm rescaling to preserve total energy
        proj_energy = result.pow(2).sum()
        if proj_energy > 0:
            scale = (src_energy / proj_energy).sqrt().clamp(1.0, 2.0)
            result = result * scale

        return result.reshape(target_shape).to(W_src.dtype)

    def _xavier_init(self, shape: tuple, dtype: torch.dtype, device: torch.device = torch.device("cpu")) -> Tensor:
        W = torch.empty(shape, dtype=dtype, device=device)
        if len(shape) >= 2:
            torch.nn.init.xavier_uniform_(W)
        else:
            torch.nn.init.uniform_(W, -0.1, 0.1)
        return W

    def pca_project(
        self,
        W_src: Tensor,
        target_shape: tuple,
        activations: Tensor,
    ) -> Tensor:
        """
        PCA-based projection for KV heads: project output dimension using
        PCA on actual model activations, then project input dimension via SVD.

        This preserves the directions that the model actually uses, rather than
        the directions with the most weight magnitude.

        W_src: (num_kv_heads * head_dim, d_model_src) e.g. (192, 576)
        target_shape: (num_kv_heads_tgt * head_dim, d_model_tgt) e.g. (128, 512)
        activations: (N, num_kv_heads * head_dim) collected k or v outputs
        """
        W_2d = W_src.float().reshape(W_src.shape[0], -1)
        out_tgt = target_shape[0]
        in_tgt = 1
        for dim in target_shape[1:]:
            in_tgt *= dim

        src_energy = W_2d.pow(2).sum()

        act = activations.float()
        act_mean = act.mean(dim=0)
        act_centered = act - act_mean
        U_act, S_act, Vh_act = torch.linalg.svd(act_centered, full_matrices=False)
        n_pca = min(out_tgt, Vh_act.shape[0])
        P = Vh_act[:n_pca, :]

        W_projected = P @ W_2d

        if W_projected.shape[1] > in_tgt:
            U2, S2, Vh2 = torch.linalg.svd(W_projected, full_matrices=False)
            k = min(len(S2), in_tgt)
            W_final = (U2[:, :k] * S2[:k].unsqueeze(0)) @ Vh2[:k, :in_tgt]
        else:
            W_final = W_projected

        proj_energy = W_final.pow(2).sum()
        if proj_energy > 0:
            scale = (src_energy / proj_energy).sqrt().clamp(1.0, 2.0)
            W_final = W_final * scale

        if W_final.shape[0] < out_tgt:
            pad = out_tgt - W_final.shape[0]
            W_final = torch.cat([W_final, torch.zeros(pad, W_final.shape[1])], dim=0)
        if W_final.shape[1] < in_tgt:
            pad = in_tgt - W_final.shape[1]
            W_final = torch.cat([W_final, torch.zeros(W_final.shape[0], pad)], dim=1)

        return W_final[:out_tgt, :in_tgt].reshape(target_shape).to(W_src.dtype)
