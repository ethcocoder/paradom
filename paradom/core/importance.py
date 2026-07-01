import torch
from torch import Tensor
from typing import Dict, Optional, Tuple
from paradom.core.weight import WeightProduct

class ImportanceScorer:
    """
    Identifies the 'Winning Tickets' — the essential weights that carry most of the intelligence.
    Uses spectral energy analysis to rank weight importance.
    """

    @staticmethod
    def svd_spectrum(
        W: Tensor,
        top_k_fraction: float = 0.20
    ) -> Tensor:
        """
        Computes an importance mask based on the singular value spectrum.
        
        Rationale: Weights contributing to the top singular values carry the maximum
        variance (intelligence). Swapping these preserves the structural core of the layer.
        """
        # Ensure 2D for SVD (handle biases or vectors if they appear)
        if W.dim() == 1:
            return torch.ones_like(W)  # Biases are usually all important
            
        # Decompose
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # Calculate cumulative energy (variance)
        energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
        
        # Find how many components are needed for the top_k_fraction energy
        # Or just use the slice of the top K singular values
        top_k = int(len(S) * top_k_fraction) + 1
        
        # Reconstruct an importance map for each weight element
        # Elements contributing most to top-k singular vectors get higher scores
        U_top = U[:, :top_k].abs().sum(dim=1).unsqueeze(1)
        Vh_top = Vh[:top_k, :].abs().sum(dim=0).unsqueeze(0)
        
        importance = U_top @ Vh_top
        
        # Normalize 0-1
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        return importance

    def score_weight(self, wp: WeightProduct, fraction: float = 0.20) -> WeightProduct:
        """Computes and assigns importance to a WeightProduct."""
        importance_map = self.svd_spectrum(wp.tensor, top_k_fraction=fraction)
        
        # Aggregate to a single scalar score for the WeightProduct (or keep as mask)
        # For now, we store the mean importance
        wp.importance_score = float(importance_map.mean())
        return wp
