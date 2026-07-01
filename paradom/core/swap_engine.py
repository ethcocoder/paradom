import torch
from typing import List, Dict, Any
from paradom.core.weight import WeightProduct
from paradom.core.equivalence import MappingScore
from paradom.math.procrustes import procrustes_projected_swap

class SwapEngine:
    """
    Executes the surgical swap of weight products.
    Implements the 'Swap, Don't Recalculate' philosophy.
    """

    def __init__(self, streaming: bool = True):
        self.streaming = streaming

    def execute_swap(
        self, 
        source_wp: WeightProduct, 
        mapping: MappingScore,
        target_shape: tuple
    ) -> torch.Tensor:
        """
        Swaps the source weight into the target position using the 
        appropriate mathematical transformation.
        """
        if mapping.swap_type == "direct":
            return self._swap_direct(source_wp.tensor, target_shape)
        
        elif mapping.swap_type == "projected":
            return self._swap_projected(source_wp.tensor, target_shape)
            
        else:
            # Fallback to projected for now
            return self._swap_projected(source_wp.tensor, target_shape)

    def _swap_direct(self, tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Direct copy swap."""
        return tensor.clone().detach()

    def _swap_projected(self, tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Projected swap using SVD/Procrustes logic."""
        return procrustes_projected_swap(tensor, shape)
