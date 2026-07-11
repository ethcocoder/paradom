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
        if top_k_fraction >= 1.0:
            return torch.ones_like(W, dtype=torch.bool)
        if top_k_fraction <= 0.0:
            return torch.zeros_like(W, dtype=torch.bool)

        if W.dim() == 1:
            # Bias or 1D weight — all equally important
            return torch.ones_like(W, dtype=torch.bool)

        # Convert to float32 for SVD stability
        W_orig_dtype = W.dtype
        W_2d = W.reshape(W.shape[0], -1).float()
        
        # Auto-switch to randomized SVD for large matrices
        if max(W_2d.shape) > 2048:
            return self.score_randomized_svd(W, top_k_fraction)

        # Compute SVD
        try:
            U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
        except RuntimeError:
            # Fallback for non-converged SVD
            return self.score_randomized_svd(W, top_k_fraction)

        # Importance score per element is derived from the outer product of
        # the singular vector magnitudes for the top singular values.
        rank = max(1, int(len(S) * top_k_fraction))
        
        # Sum of absolute magnitudes for top 'rank' singular vectors
        u_importance = U[:, :rank].abs().sum(dim=1, keepdim=True)
        vh_importance = Vh[:rank, :].abs().sum(dim=0, keepdim=True)
        
        importance = (u_importance @ vh_importance).reshape(W.shape)

        # Create mask: top k_fraction% by importance score
        if importance.numel() > 16000000:
            import numpy as np
            threshold = torch.tensor(np.quantile(importance.cpu().float().numpy(), 1 - top_k_fraction), device=importance.device, dtype=importance.dtype)
        else:
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
        w_abs = W.abs()
        if w_abs.numel() > 16000000:
            import numpy as np
            threshold = torch.tensor(np.quantile(w_abs.cpu().float().numpy(), sparsity), device=W.device, dtype=W.dtype)
        else:
            threshold = w_abs.flatten().quantile(sparsity)
        return w_abs >= threshold
    def score_randomized_svd(
        self,
        W: Tensor,
        top_k_fraction: float = 0.20,
        n_iter: int = 4
    ) -> Tensor:
        """
        Faster SVD-based importance for large matrices using Randomized SVD.
        Recommended for weights > 2048 in any dimension.
        """
        if top_k_fraction >= 1.0:
            return torch.ones_like(W, dtype=torch.bool)
        if top_k_fraction <= 0.0:
            return torch.zeros_like(W, dtype=torch.bool)

        if W.dim() == 1:
            return torch.ones_like(W, dtype=torch.bool)

        W_2d = W.reshape(W.shape[0], -1).float()
        
        # Determine target rank for the randomized projection
        k = max(1, int(min(W_2d.shape) * top_k_fraction * 2)) # Buffering rank
        k = min(k, min(W_2d.shape))

        # Native PyTorch low-rank SVD
        U, S, V = torch.svd_lowrank(W_2d, q=k, niter=n_iter)
        Vh = V.t()

        rank = max(1, int(len(S) * 0.5)) # We already took a subset 'k'
        
        u_importance = U[:, :rank].abs().sum(dim=1, keepdim=True)
        vh_importance = Vh[:rank, :].abs().sum(dim=0, keepdim=True)
        importance = (u_importance @ vh_importance).reshape(W.shape)

        if importance.numel() > 16000000:
            import numpy as np
            threshold = torch.tensor(np.quantile(importance.cpu().float().numpy(), 1 - top_k_fraction), device=importance.device, dtype=importance.dtype)
        else:
            threshold = importance.flatten().quantile(1 - top_k_fraction)
        return importance >= threshold
