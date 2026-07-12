import torch
from torch import Tensor
from typing import Optional
from .enums import SwapType

class SwapEngine:
    """
    Executes the actual weight swap operations.
    """

    def swap(
        self,
        source_weight: Tensor,
        target_shape: tuple,
        swap_type: SwapType,
        importance_mask: Optional[Tensor] = None,
        axis_labels: Optional[tuple] = None
    ) -> Tensor:
        """Execute a swap operation."""
        if swap_type == SwapType.DIRECT:
            return self._direct_swap(source_weight, target_shape, importance_mask)
        elif swap_type == SwapType.PROJECTED:
            return self._projected_swap(source_weight, target_shape, importance_mask)
        elif swap_type == SwapType.OT:
            return self._ot_swap(source_weight, target_shape, importance_mask, axis_labels)
        elif swap_type == SwapType.SKIP:
            return self._xavier_init(target_shape, source_weight.dtype)
        else:
            raise NotImplementedError(f"Swap type {swap_type} not yet implemented for Phase 1.")

    def _direct_swap(
        self,
        W_src: Tensor,
        target_shape: tuple,
        mask: Optional[Tensor]
    ) -> Tensor:
        if W_src.shape != target_shape:
            raise ValueError(f"Direct swap requires identical shapes: {W_src.shape} vs {target_shape}")

        if mask is None:
            return W_src.clone().detach()

        # Only swap masked elements, initialize rest with Xavier
        W_target = self._xavier_init(target_shape, W_src.dtype, W_src.device)
        
        # Ensure mask and weights are on same device/dtype
        mask = mask.to(W_src.device)
        W_target[mask] = W_src[mask]
        return W_target

    def _projected_swap(
        self,
        W_src: Tensor,
        target_shape: tuple,
        mask: Optional[Tensor]
    ) -> Tensor:
        """Project source weight to target shape via SVD truncation."""
        if mask is not None and mask.shape == W_src.shape:
            # Apply sparsity mask before calculating the SVD spectrum
            W_src = W_src.clone()
            W_src[~mask] = 0.0

        # Flatten source to 2D
        W_2d = W_src.reshape(W_src.shape[0], -1).float()
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

        # Target dimensions
        d_out_tgt = target_shape[0]
        # Calculate target input dimension (handle multi-dim targets by flattening rest)
        d_in_tgt = 1
        for dim in target_shape[1:]:
            d_in_tgt *= dim
        
        rank = min(len(S), d_out_tgt, d_in_tgt)

        # Truncate singular matrices
        U_r  = U[:d_out_tgt, :rank]
        S_r  = S[:rank]
        Vh_r = Vh[:rank, :d_in_tgt]

        # Reconstruct in target 2D shape
        W_reconstructed = (U_r * S_r.unsqueeze(0)) @ Vh_r
        
        # Reshape to target actual shape
        W_target_2d = torch.zeros((d_out_tgt, d_in_tgt), dtype=torch.float32, device=W_src.device)
        r_out = min(d_out_tgt, W_reconstructed.shape[0])
        r_in = min(d_in_tgt, W_reconstructed.shape[1])
        W_target_2d[:r_out, :r_in] = W_reconstructed[:r_out, :r_in]
        
        W_target = W_target_2d.reshape(target_shape).to(W_src.dtype)
        
        # If mask was target shaped, apply post-projection
        if mask is not None and mask.shape == target_shape:
            W_final = self._xavier_init(target_shape, W_src.dtype, W_src.device)
            W_final[mask] = W_target[mask]
            return W_final
        
        return W_target

    def _ot_swap(self, W_src: Tensor, target_shape: tuple, mask: Optional[Tensor], axis_labels: Optional[tuple] = None) -> Tensor:
        """
        Shared Spectral Projection for downscaling.
        
        Instead of independently merging vectors per-layer (which breaks cross-layer
        compatibility), this computes a SHARED orthogonal projection basis via SVD
        of the first weight matrix needing each (src_dim → tgt_dim) reduction.
        
        All subsequent layers reuse the same projector, guaranteeing that
        layer N's output space == layer N+1's input space.
        
        For W ∈ R^(m×n) → R^(m'×n'):
          - If n > n': W_new = W @ P       where P ∈ R^(n×n') is a shared right projector
          - If m > m': W_new = Q^T @ W_new  where Q ∈ R^(m×m') is a shared left projector
        """
        # Target flat shapes
        d_out_tgt = target_shape[0]
        d_in_tgt = 1
        for dim in target_shape[1:]: d_in_tgt *= dim
        W_src_d_in = W_src.numel() // W_src.shape[0]

        # OT is purely a condensation (downscaling) algorithm. Upscaling falls back to SVD padding.
        if W_src.shape[0] < d_out_tgt or W_src_d_in < d_in_tgt:
            return self._projected_swap(W_src, target_shape, mask)
        
        W_2d = W_src.reshape(W_src.shape[0], -1).float()
        result = W_2d
        
        # Project input dimension (columns) if needed
        if W_src_d_in > d_in_tgt:
            label = axis_labels[1] if axis_labels and len(axis_labels) > 1 else None
            P = self._get_spectral_projector(W_src_d_in, d_in_tgt, 'right', W_2d, label)
            result = result @ P  # (m, n) @ (n, n') → (m, n')
        
        # Project output dimension (rows) if needed
        # Use the CURRENT result (possibly column-projected) for the left projector basis,
        # preventing diagonal collapse when both projectors come from the same SVD.
        basis = result if W_src_d_in > d_in_tgt else W_2d
        if W_src.shape[0] > d_out_tgt:
            label = axis_labels[0] if axis_labels else None
            Q = self._get_spectral_projector(W_src.shape[0], d_out_tgt, 'left', basis, label)
            result = Q.T @ result  # (m', m) @ (m, n') → (m', n')
        
        return result.reshape(target_shape).to(W_src.dtype)

    def _get_spectral_projector(self, src_dim: int, tgt_dim: int, direction: str, W_2d: Tensor, axis_label: Optional[str] = None) -> Tensor:
        """
        Get or compute a shared spectral projection matrix.
        
        The projector is computed via SVD of the FIRST weight matrix that needs this
        specific (src_dim → tgt_dim) reduction. All subsequent layers reuse the same
        basis, preserving cross-layer topological alignment.
        
        Uses full_matrices=True when tgt_dim > rank to obtain a complete
        orthogonal basis (needed for FFN dim reductions like 1536→1408).
        """
        if not hasattr(self, '_spectral_cache'):
            self._spectral_cache = {}
        
        cache_key = (axis_label, direction) if axis_label is not None else (src_dim, tgt_dim, direction)
        
        if cache_key not in self._spectral_cache:
            rank = min(W_2d.shape)
            need_full = tgt_dim > rank
            
            if direction == 'right':
                # Project columns: W @ P  where P = V[:, :tgt_dim]
                _, _, Vh = torch.linalg.svd(W_2d.float(), full_matrices=need_full)
                P = Vh[:tgt_dim, :].T  # (src_dim, tgt_dim)
                self._spectral_cache[cache_key] = P
            else:
                # Project rows: Q^T @ W  where Q = U[:, :tgt_dim]
                U, _, _ = torch.linalg.svd(W_2d.float(), full_matrices=need_full)
                Q = U[:, :tgt_dim]  # (src_dim, tgt_dim)
                self._spectral_cache[cache_key] = Q
        
        return self._spectral_cache[cache_key]

    def _xavier_init(self, shape: tuple, dtype: torch.dtype, device: torch.device = torch.device("cpu")) -> Tensor:
        W = torch.empty(shape, dtype=dtype, device=device)
        if len(shape) >= 2:
            torch.nn.init.xavier_uniform_(W)
        else:
            # 1D tensors (biases, etc)
            torch.nn.init.uniform_(W, -0.1, 0.1)
        return W
