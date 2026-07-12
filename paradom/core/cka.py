"""Linear CKA similarity for weight equivalence scoring."""

import torch
from torch import Tensor


def linear_cka(X: Tensor, Y: Tensor) -> float:
    """
    Compute linear Centered Kernel Alignment between two feature matrices.

    Each matrix is treated as (n_samples, n_features). For weight tensors of
    different shapes, callers should reshape to 2D with matching column count
    or use ``weight_cka`` which handles projection internally.
    """
    if X.numel() == 0 or Y.numel() == 0:
        return 0.0

    X = X.reshape(X.shape[0], -1).float()
    Y = Y.reshape(Y.shape[0], -1).float()

    col = min(X.shape[1], Y.shape[1])
    X = X[:, :col]
    Y = Y[:, :col]

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = torch.linalg.norm(Y.T @ X) ** 2
    hsic_xx = torch.linalg.norm(X.T @ X) ** 2
    hsic_yy = torch.linalg.norm(Y.T @ Y) ** 2

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom <= 1e-12:
        return 0.0
    return float((hsic_xy / denom).clamp(0.0, 1.0).item())


def weight_cka(source: Tensor, target: Tensor) -> float:
    """
    CKA between two weight tensors, projecting to a shared 2D representation.
    """
    src = source.reshape(source.shape[0], -1).float()
    tgt = target.reshape(target.shape[0], -1).float()

    row = min(src.shape[0], tgt.shape[0])
    col = min(src.shape[1], tgt.shape[1])
    return linear_cka(src[:row, :col], tgt[:row, :col])
