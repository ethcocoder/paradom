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
        importance_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Execute a swap operation."""
        if swap_type == SwapType.DIRECT:
            return self._direct_swap(source_weight, target_shape, importance_mask)
        elif swap_type == SwapType.PROJECTED:
            return self._projected_swap(source_weight, target_shape, importance_mask)
        elif swap_type == SwapType.SKIP:
            return self._xavier_init(target_shape, source_weight.dtype)
        else:
            # For Phase 1, we limit to DIRECT and PROJECTED
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
        W_target = self._xavier_init(target_shape, W_src.dtype)
        
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
        W_target_2d = torch.zeros((d_out_tgt, d_in_tgt), dtype=torch.float32)
        r_out = min(d_out_tgt, W_reconstructed.shape[0])
        r_in = min(d_in_tgt, W_reconstructed.shape[1])
        W_target_2d[:r_out, :r_in] = W_reconstructed[:r_out, :r_in]
        
        W_target = W_target_2d.reshape(target_shape).to(W_src.dtype)
        
        if mask is not None:
            # If a mask is provided, we only keep the projected weights 
            # where the mask indicates importance. 
            # Note: Mask should ideally be in target shape if provided for projected swap.
            # For Phase 1 simplification, we'll assume mask is for source or same-arch.
            # If shape mismatch, we skip masking for now or re-project the mask.
            if mask.shape == target_shape:
                W_final = self._xavier_init(target_shape, W_src.dtype)
                W_final[mask] = W_target[mask]
                return W_final
        
        return W_target

    def _xavier_init(self, shape: tuple, dtype: torch.dtype) -> Tensor:
        W = torch.empty(shape, dtype=dtype)
        if len(shape) >= 2:
            torch.nn.init.xavier_uniform_(W)
        else:
            # 1D tensors (biases, etc)
            torch.nn.init.uniform_(W, -0.1, 0.1)
        return W
