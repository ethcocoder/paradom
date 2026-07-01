import torch
from torch import Tensor

def linear_cka(X: Tensor, Y: Tensor) -> float:
    """
    Computes Linear CKA (Centered Kernel Alignment) between two activation matrices.
    
    CKA is invariant to orthogonal transformation and isotropic scaling,
    making it ideal for comparing representations across different architectures.
    
    Formula: CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    """
    # X and Y shapes: (n_samples, n_features)
    # n_samples must be the same for both
    
    # Center the matrices
    def center(K):
        n = K.shape[0]
        unit = torch.ones(n, n) / n
        return K - unit @ K - K @ unit + unit @ K @ unit

    K = X @ X.T
    L = Y @ Y.T
    
    K_centered = center(K)
    L_centered = center(L)
    
    # HSIC is Frobenius inner product of centered kernels
    hsic_kl = (K_centered * L_centered).sum()
    hsic_kk = (K_centered * K_centered).sum()
    hsic_ll = (L_centered * L_centered).sum()
    
    score = hsic_kl / torch.sqrt(hsic_kk * hsic_ll + 1e-8)
    return float(score)

def cka_from_features(X: Tensor, Y: Tensor) -> float:
    """Fast version of linear CKA using Frobenius norm directly."""
    # Source: https://arxiv.org/abs/1905.00414
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    dot_product_norm = torch.linalg.norm(X.T @ Y, ord='fro')**2
    norm_x = torch.linalg.norm(X.T @ X, ord='fro')
    norm_y = torch.linalg.norm(Y.T @ Y, ord='fro')
    
    score = dot_product_norm / (norm_x * norm_y + 1e-8)
    return float(score)
