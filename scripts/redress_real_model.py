import numpy as np
from safetensors.numpy import save_file, load_file
import re
import os

def procrustes_project(W_src, target_shape):
    rows_tgt, cols_tgt = target_shape
    # Handle vectors (1D)
    if len(W_src.shape) == 1:
        new_v = np.zeros(target_shape, dtype=W_src.dtype)
        r = min(len(W_src), rows_tgt)
        new_v[:r] = W_src[:r]
        return new_v
        
    # Standard 2D SVD Projection
    rows_src, cols_src = W_src.shape
    U, S, Vh = np.linalg.svd(W_src, full_matrices=False)
    r = min(len(S), rows_tgt, cols_tgt)
    U_r = U[:rows_tgt, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :cols_tgt]
    W_projected = (U_r * S_r[np.newaxis, :]) @ Vh_r
    final = np.zeros(target_shape, dtype=W_src.dtype)
    final[:W_projected.shape[0], :W_projected.shape[1]] = W_projected
    return final

def identify_role(name):
    if "q_proj" in name: return "CONTEXT_QUERY"
    if "k_proj" in name: return "CONTEXT_KEY"
    if "v_proj" in name: return "CONTEXT_VALUE"
    if "o_proj" in name: return "CONTEXT_OUTPUT"
    if "gate_proj" in name: return "FFN_EXPAND"
    if "up_proj" in name: return "FFN_EXPAND"
    if "down_proj" in name: return "FFN_CONTRACT"
    return None

def redress_real():
    print("STARTING: Real-World Redress (SmolLM-135M -> Sovereign-Master-32)...")
    source_weights = load_file("models/hf_source/model.safetensors")
    target_weights = {}
    
    # 1. Group source weights by role and layer
    layer_intelligence = {} # {role: [tensors]}
    for name, tensor in source_weights.items():
        role = identify_role(name)
        if role:
            if role not in layer_intelligence: layer_intelligence[role] = []
            layer_intelligence[role].append(tensor)
            
    print(f"  Captured intelligence for roles: {list(layer_intelligence.keys())}")

    # 2. Map into 32 Target Layers
    # We recycle the intelligence from the 30 layers to fill the 32 slots 
    # (The last 2 layers receive intelligence from the mirrored upper layers)
    for i in range(32):
        src_idx = i % 30 # Wrap around for the last 2 expansion layers
        
        # Define target dimensions for Master-32
        # (640 hidden, 1728 FFN)
        specs = {
            "query":    (640, 640),
            "key":      (640, 640),
            "value":    (640, 640),
            "out":      (640, 640),
            "ffn_up":   (1728, 640),
            "ffn_down": (640, 1728)
        }
        
        # A. Attention
        target_weights[f"layers.{i}.attention.query"] = procrustes_project(layer_intelligence["CONTEXT_QUERY"][src_idx], specs["query"])
        target_weights[f"layers.{i}.attention.key"]   = procrustes_project(layer_intelligence["CONTEXT_KEY"][src_idx], specs["key"])
        target_weights[f"layers.{i}.attention.value"] = procrustes_project(layer_intelligence["CONTEXT_VALUE"][src_idx], specs["value"])
        target_weights[f"layers.{i}.attention.out"]   = procrustes_project(layer_intelligence["CONTEXT_OUTPUT"][src_idx], specs["out"])
        
        # B. FFN (SmolLM uses gate/up and down)
        # We fuse gate and up into our single 'expand' role for this master test
        target_weights[f"layers.{i}.ffn.expand"]   = procrustes_project(layer_intelligence["FFN_EXPAND"][src_idx*2], specs["ffn_up"])
        target_weights[f"layers.{i}.ffn.contract"] = procrustes_project(layer_intelligence["FFN_CONTRACT"][src_idx], specs["ffn_down"])
        
        if i % 8 == 0: print(f"  Processed Layer {i}/32...")

    # 3. Finalize
    os.makedirs("models/sovereign_master", exist_ok=True)
    save_file(target_weights, "models/sovereign_master/model.safetensors")
    print(f"\nSUCCESS: Sovereign-Master-32 Created.")
    print(f"📏 Final Parameters: {sum(t.size for t in target_weights.values()):,}")

if __name__ == "__main__":
    redress_real()
