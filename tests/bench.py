import torch
import time
from paradom.core.importance import ImportanceScorer

def test_randomized_svd_speed():
    scorer = ImportanceScorer()
    # Large matrix (4096, 4096)
    W = torch.randn(4096, 4096)
    
    print("\n--- Benchmark: SVD on 4096x4096 Tensor ---")
    
    # Test Exact
    print("Running Exact SVD...")
    t0 = time.perf_counter()
    # Note: score_svd_spectrum will call randomized if > 2048, 
    # so we should use torch.linalg.svd directly for exact comparison
    # or modify ImportanceScorer to allow forcing exact.
    
    W_2d = W.reshape(W.shape[0], -1).float()
    U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
    t_exact = time.perf_counter() - t0
    print(f"Exact SVD: {t_exact:.4f}s")
    
    # Test Randomized
    print("Running Randomized SVD...")
    t0 = time.perf_counter()
    mask_rand = scorer.score_randomized_svd(W, top_k_fraction=0.10)
    t_rand = time.perf_counter() - t0
    print(f"Randomized SVD: {t_rand:.4f}s")
    
    speedup = t_exact / t_rand
    print(f"Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    test_randomized_svd_speed()
