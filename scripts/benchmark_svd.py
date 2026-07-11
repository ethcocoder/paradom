import time
import torch
import numpy as np

# Adjust python path if necessary or run from root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paradom.core.importance import ImportanceScorer

def benchmark():
    scorer = ImportanceScorer()
    shapes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 4096)
    ]
    
    print(f"{'Shape':<15} | {'Exact SVD (s)':<15} | {'Randomized SVD (s)':<15}")
    print("-" * 50)
    
    for shape in shapes:
        W = torch.randn(*shape)
        
        # Exact SVD (bypassing the 2048 safeguard to force exact computation)
        # We'll just run the exact svd logic manually here to get pure timings.
        W_2d = W.reshape(W.shape[0], -1).float()
        
        t0 = time.time()
        try:
            U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
        except RuntimeError:
            pass
        exact_time = time.time() - t0
        
        # Randomized SVD
        # We can just time the underlying randomized_svd similarly, or time the scorer function
        t0 = time.time()
        scorer.score_randomized_svd(W)
        rand_time = time.time() - t0
        
        print(f"{str(shape):<15} | {exact_time:<15.4f} | {rand_time:<15.4f}")

if __name__ == '__main__':
    benchmark()
