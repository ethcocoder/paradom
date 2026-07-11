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
        elif swap_type == SwapType.OT:
            return self._ot_swap(source_weight, target_shape, importance_mask)
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

    def _ot_swap(self, W_src: Tensor, target_shape: tuple, mask: Optional[Tensor]) -> Tensor:
        """
        Optimal Transport Barycenter / Cosine Merging.
        Condenses a larger matrix into a smaller one by mathematically fusing redundant feature vectors.
        """
        if W_src.dim() < 2: 
            return self._projected_swap(W_src, target_shape, mask)
            
        # Target flat shapes
        d_out_tgt = target_shape[0]
        d_in_tgt = 1
        for dim in target_shape[1:]: d_in_tgt *= dim
        W_src_d_in = W_src.numel() // W_src.shape[0]

        # OT is purely a condensation (downscaling) algorithm. Upscaling should safely fallback to SVD padding.
        if W_src.shape[0] < d_out_tgt or W_src_d_in < d_in_tgt:
            return self._projected_swap(W_src, target_shape, mask)
            
        W_out = W_src.clone()
        W_out = self._condense_dim(W_out, d_out_tgt, 0)
        
        W_out_2d = W_out.reshape(W_out.shape[0], -1)
        W_out_2d = self._condense_dim(W_out_2d, d_in_tgt, 1)
        
        return W_out_2d.reshape(target_shape).to(W_src.dtype)

    def _condense_dim(self, tensor: Tensor, target_size: int, dim: int) -> Tensor:
        """
        Condenses `tensor` along `dim` down to `target_size` by fusing closest vectors.
        Uses pairwise Cosine Similarity grouping, batched for immediate execution.
        """
        current_size = tensor.shape[dim]
        if current_size <= target_size: return tensor
        
        num_merges = current_size - target_size
        t = tensor.transpose(0, dim).clone() # Shape: [C, F]
        
        t_flat = t.reshape(t.shape[0], -1).float()
        
        # L2 Normalize for Cosine Similarity
        norms = torch.norm(t_flat, p=2, dim=1, keepdim=True).clamp_min(1e-8)
        t_norm = t_flat / norms
        sim = torch.mm(t_norm, t_norm.t()) # [C, C]
        sim.fill_diagonal_(-float('inf'))
        
        # Only take top K indices (avoid sorting all 2+ million elements and unbind overhead)
        k = min(sim.numel(), num_merges * 10)
        _, top_k_indices = torch.topk(sim.flatten(), k)
        
        merged_rows = set()
        pairs_to_merge = []
        
        # Convert to Python list immediately for ultra-fast iteration
        for idx in top_k_indices.tolist():
            if len(pairs_to_merge) >= num_merges: break
            
            r1 = idx // sim.shape[0]
            r2 = idx % sim.shape[0]
            
            if r1 != r2 and r1 not in merged_rows and r2 not in merged_rows:
                pairs_to_merge.append((min(r1, r2), max(r1, r2)))
                merged_rows.add(r1)
                merged_rows.add(r2)
                
        rows_to_drop = []
        for r1, r2 in pairs_to_merge:
            m1 = norms[r1].item()
            m2 = norms[r2].item()
            w_sum = m1 + m2 + 1e-8
            
            # Magnitude-weighted Barycenter merge
            t[r1] = (t[r1] * (m1/w_sum)) + (t[r2] * (m2/w_sum))
            rows_to_drop.append(r2)
            
        # Drop merged residuals
        keep_mask = torch.ones(t.shape[0], dtype=torch.bool, device=t.device)
        keep_mask[rows_to_drop] = False
        t = t[keep_mask]
        
        t = t.transpose(0, dim).to(tensor.dtype)
        
        # Recurse if we couldn't find enough disjoint pairs in one pass (rare)
        if t.shape[dim] > target_size:
            return self._condense_dim(t, target_size, dim)
            
        return t

    def _xavier_init(self, shape: tuple, dtype: torch.dtype, device: torch.device = torch.device("cpu")) -> Tensor:
        W = torch.empty(shape, dtype=dtype, device=device)
        if len(shape) >= 2:
            torch.nn.init.xavier_uniform_(W)
        else:
            # 1D tensors (biases, etc)
            torch.nn.init.uniform_(W, -0.1, 0.1)
        return W
