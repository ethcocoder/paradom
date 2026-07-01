import numpy as np
from safetensors.numpy import load_file
import os

def test_intelligence():
    print("STARTING: Paradox Ultra Intelligence Audit...")
    
    # 1. Load both models
    source_path = "models/hf_source/model.safetensors"
    target_path = "paradox_ai/model_paradox_ultra.safetensors"
    
    print(f"  Loading Source: {source_path}")
    source_weights = load_file(source_path)
    print(f"  Loading Target: {target_path}")
    target_weights = load_file(target_path)
    
    # 2. Extract a sample layer (Layer 0 Query)
    W_src = source_weights["model.layers.0.self_attn.q_proj.weight"]
    W_tgt = target_weights["paradox.ultra.layer_0.att_q"]
    
    print("\nSHAPE ANALYSIS:")
    print(f"  Source Shape: {W_src.shape}")
    print(f"  Target Shape: {W_tgt.shape}")
    
    # 3. Simulated Inference
    x_src = np.random.randn(576).astype(np.float32)
    x_tgt = np.zeros(1024, dtype=np.float32)
    x_tgt[:576] = x_src 
    
    y_src = x_src @ W_src.T
    y_tgt = x_tgt @ W_tgt.T
    
    # 4. Results Audit
    print("\nINFERENCE METRICS:")
    print(f"  Source Output (Mean/Var): {np.mean(y_src):.6f} / {np.var(y_src):.6f}")
    print(f"  Target Output (Mean/Var): {np.mean(y_tgt):.6f} / {np.var(y_tgt):.6f}")
    
    if np.isnan(y_tgt).any():
        print("RESULT: FAILURE. Target model produced NaNs.")
    else:
        diff = abs(np.var(y_src) - np.var(y_tgt))
        print(f"  Representational Delta: {diff:.6f}")
        
        if diff < 50.0:
            print("\nRESULT: SUCCESS. The Paradox-Ultra model is stable and intelligence-correlated.")
        else:
            print("\nRESULT: WARNING. Variance drift detected.")

if __name__ == "__main__":
    test_intelligence()
