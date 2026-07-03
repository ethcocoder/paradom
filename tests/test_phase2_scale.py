import torch
import time
import os
import psutil
from paradom.core.importance import ImportanceScorer
from paradom.core.engine import Paradom
from paradom.mappings.transformer_to_mamba import TransformerToMambaMapper
from paradom.core.types import WeightProduct
from paradom.core.enums import FunctionalRole

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_randomized_svd_speed():
    scorer = ImportanceScorer()
    # Large matrix (4096, 4096)
    W = torch.randn(4096, 4096)
    
    print("\n--- Benchmark: SVD on 4096x4096 Tensor ---")
    
    # Test Exact
    t0 = time.perf_counter()
    mask_exact = scorer.score_svd_spectrum(W, top_k_fraction=0.10)
    t_exact = time.perf_counter() - t0
    print(f"Exact SVD: {t_exact:.4f}s")
    
    # Test Randomized
    t0 = time.perf_counter()
    mask_rand = scorer.score_randomized_svd(W, top_k_fraction=0.10)
    t_rand = time.perf_counter() - t0
    print(f"Randomized SVD: {t_rand:.4f}s")
    
    speedup = t_exact / t_rand
    print(f"Speedup: {speedup:.1f}x")
    
    # Verify overlap (Randomized should capture similar top-k regions)
    intersection = (mask_exact & mask_rand).float().sum()
    union = (mask_exact | mask_rand).float().sum()
    iou = intersection / union
    print(f"Mask IoU (overlap): {iou:.4f}")
    assert speedup > 2.0
    assert iou > 0.1 # Should have some overlap at least

def test_streaming_ram_efficiency():
    print("\n--- Benchmark: Streaming RAM Efficiency ---")
    initial_ram = get_ram_usage()
    print(f"Initial RAM: {initial_ram:.2f}MB")
    
    # 1. Create a dummy model on disk
    # (Simulate a layer with 4096x4096 weights ~64MB per tensor)
    dummy_path = "output/dummy_7b.pt"
    dummy_model = {
        f"layers.{i}.q_proj.weight": torch.randn(4096, 4096)
        for i in range(10) # 640MB of weights
    }
    torch.save(dummy_model, dummy_path)
    
    post_disk_ram = get_ram_usage()
    print(f"RAM after disk save: {post_disk_ram:.2f}MB")
    
    # 2. Run streaming swap
    paradom = Paradom()
    # Mock config
    config = {"d_model": 4096, "d_inner": 8192}
    
    t0 = time.perf_counter()
    report = paradom.stream_swap(
        dummy_path,
        target_architecture="mamba",
        target_config=config,
        source_architecture="llama",
        output_path="output/swapped_streaming"
    )
    t_swap = time.perf_counter() - t0
    
    final_ram = get_ram_usage()
    peak_ram_diff = final_ram - initial_ram
    
    print(f"Swap Time: {t_swap:.2f}s")
    print(f"Final RAM: {final_ram:.2f}MB")
    print(f"Estimated RAM overhead: {peak_ram_diff:.2f}MB")
    
    # In streaming mode, we shouldn't have loaded all layers at once
    # So RAM shouldn't have spiked to 640MB+
    assert peak_ram_diff < 500 # Adjust based on environment

if __name__ == "__main__":
    test_randomized_svd_speed()
    test_streaming_ram_efficiency()
    print("\n✅ Phase 2 Month 4 Infrastructure Tests PASSED")
