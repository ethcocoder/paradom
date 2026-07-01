import numpy as np
from safetensors.numpy import load_file, save_file

class ParadoxFoundation:
    """The stable math foundation for Paradom redressing in zero-dependency environments."""
    
    @staticmethod
    def procrustes_project(W_src, target_shape):
        """Projects source intelligence onto target dimensions (1D or 2D).
        
        CRITICAL: When source and target shapes match, we perform a DIRECT SWAP.
        This is the core Paradom principle — identical numerical products need 
        no mathematical transformation. Only mismatched dimensions require SVD projection.
        """
        # FAST PATH: Direct Equivalence Swap (shapes match exactly)
        if W_src.shape == target_shape:
            return W_src.copy()
        
        # Handle 1D Vectors (e.g. Norms, Biases)
        if len(target_shape) == 1:
            size_tgt = target_shape[0]
            final = np.zeros((size_tgt,), dtype=W_src.dtype)
            r = min(len(W_src), size_tgt)
            final[:r] = W_src[:r]
            return final
            
        rows_tgt, cols_tgt = target_shape
        if len(W_src.shape) == 1:
            new_v = np.zeros(target_shape, dtype=W_src.dtype)
            r = min(len(W_src), rows_tgt)
            new_v[:r] = W_src[:r]
            return new_v
            
        U, S, Vh = np.linalg.svd(W_src, full_matrices=False)
        r = min(len(S), rows_tgt, cols_tgt)
        U_r, S_r, Vh_r = U[:rows_tgt, :r], S[:r], Vh[:r, :cols_tgt]
        W_projected = (U_r * S_r[np.newaxis, :]) @ Vh_r
        
        final = np.zeros(target_shape, dtype=W_src.dtype)
        final[:W_projected.shape[0], :W_projected.shape[1]] = W_projected
        return final

    @staticmethod
    def load_intelligence(path):
        """Loads a model shard into memory."""
        return load_file(path)

    @staticmethod
    def save_redressed(weights, path):
        """Persists the newly redressed model."""
        save_file(weights, path)
