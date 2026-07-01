import torch
from torch import Tensor
from typing import Tuple

def orthogonal_procrustes(A: Tensor, B: Tensor) -> Tensor:
    """
    Solves the Orthogonal Procrustes problem:
    Finds an orthogonal matrix R that minimizes ||A @ R - B||_F.
    
    Used to align Weight Space A to Weight Space B.
    """
    # A and B must have the same shape
    M = A.T @ B
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    return R

def procrustes_projected_swap(W_source: Tensor, target_shape: Tuple[int, ...]) -> Tensor:
    """
    Projects source weights into target dimensions using SVD-based alignment.
    
    This is the core of the 'Projected Swap' (3=4-1 logic).
    It extracts the top principal components from the source and 
    maps them into the target's coordinate system.
    """
    # Handle 2D weights for now (standard linear layers)
    if W_source.dim() != 2:
        # Fallback for vectors or high-dim tensors (padding/truncating)
        return _simple_reshape(W_source, target_shape)

    rows_src, cols_src = W_source.shape
    rows_tgt, cols_tgt = target_shape

    # Decompose source to find maximum variance directions (intelligence)
    U, S, Vh = torch.linalg.svd(W_source, full_matrices=False)
    
    # Truncate or pad singular values to target shape
    r = min(len(S), rows_tgt, cols_tgt)
    
    # Extract the top 'r' components (the Winning Ticket)
    U_r = U[:rows_tgt, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :cols_tgt]
    
    # Reconstruct in target shape
    W_projected = (U_r * S_r.unsqueeze(0)) @ Vh_r
    
    # If target is larger, zero-pad the rest
    if rows_tgt > rows_src or cols_tgt > cols_src:
        final = torch.zeros(target_shape, dtype=W_source.dtype, device=W_source.device)
        final[:W_projected.shape[0], :W_projected.shape[1]] = W_projected
        return final
        
    return W_projected

def _simple_reshape(W: Tensor, target_shape: Tuple[int, ...]) -> Tensor:
    """Simple resize with zero-padding or truncation."""
    new_w = torch.zeros(target_shape, dtype=W.dtype, device=W.device)
    
    # Determine the slice to copy
    slices = tuple(slice(0, min(s, t)) for s, t in zip(W.shape, target_shape))
    new_w[slices] = W[slices]
    return new_w
