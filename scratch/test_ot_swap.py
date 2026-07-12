import torch
import sys
import os
import time

# Add to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paradom.core.swap_engine import SwapEngine

def test():
    engine = SwapEngine()
    print("Testing _ot_swap...")
    
    # 576 -> 512
    print("Testing 576 -> 512...")
    t0 = time.time()
    w1 = torch.randn(576, 576)
    out1 = engine._ot_swap(w1, (512, 512), None)
    print(f"out1 shape: {out1.shape}, time: {time.time()-t0:.2f}s")
    
    print("Testing 49152 x 576 -> 49152 x 512 (embedding)...")
    t0 = time.time()
    w2 = torch.randn(49152, 576)
    out2 = engine._ot_swap(w2, (49152, 512), None)
    print(f"out2 shape: {out2.shape}, time: {time.time()-t0:.2f}s")

    print("Testing 1536 x 576 -> 1408 x 512 (FFN gate_proj)...")
    t0 = time.time()
    w3 = torch.randn(1536, 576)
    out3 = engine._ot_swap(w3, (1408, 512), None)
    print(f"out3 shape: {out3.shape}, time: {time.time()-t0:.2f}s")
    
    print("All tests passed.")

if __name__ == '__main__':
    test()
