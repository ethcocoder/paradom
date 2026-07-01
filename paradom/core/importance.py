import torch
from torch import Tensor
from typing import Optional

class ImportanceScorer:
    """
    Identifies which weights carry the essential intelligence.
    Only these weights are swapped — the rest are initialized fresh.
    This is the "winning ticket" finder.
    """

    def score_svd_spectrum(
        self,
        W: Tensor,
        top_k_fraction: float = 0.20
    ) -> Tensor:
        """
        Returns a boolean mask of the top-k% most important weights
        by their contribution to the singular value spectrum.
        """
        if W.dim() == 1:
            # Bias or 1D weight — all equally important
            return torch.ones_like(W, dtype=torch.bool)

        # Convert to float32 for SVD stability
        W_orig_dtype = W.dtype
        W_2d = W.reshape(W.shape[0], -1).float()
        
        # Compute SVD
        try:
            U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
        except RuntimeError:
            # Fallback for non-converged SVD
            return torch.ones_like(W, dtype=torch.bool)

        # Importance score per element is derived from the outer product of
        # the singular vector magnitudes for the top singular values.
        rank = max(1, int(len(S) * top_k_fraction))
        
        # Sum of absolute magnitudes for top 'rank' singular vectors
        u_importance = U[:, :rank].abs().sum(dim=1, keepdim=True)
        vh_importance = Vh[:rank, :].abs().sum(dim=0, keepdim=True)
        
        importance = (u_importance @ vh_importance).reshape(W.shape)

        # Create mask: top k_fraction% by importance score
        threshold = importance.flatten().quantile(1 - top_k_fraction)
        return importance >= threshold

    def score_lottery_ticket(
        self,
        W: Tensor,
        sparsity: float = 0.80
    ) -> Tensor:
        """
        Classic magnitude-based importance: keep top (1-sparsity)% by |W|.
        """
        threshold = W.abs().flatten().quantile(sparsity)
        return W.abs() >= threshold
