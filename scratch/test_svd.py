import torch
import time

def benchmark():
    W = torch.randn(8192, 4096)
    t0 = time.time()
    
    # Target rank calculation
    q = int(min(W.shape) * 0.20 * 2)
    q = min(q, min(W.shape))
    
    U, S, V = torch.svd_lowrank(W, q=q, niter=4)
    print(f"Native PyTorch svd_lowrank: {time.time()-t0:.4f} s")
    print(f"U: {U.shape}, S: {S.shape}, V: {V.shape}")

if __name__ == '__main__':
    benchmark()
