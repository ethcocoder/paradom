import numpy as np
from safetensors.numpy import save_file, load_file
import yaml
import os

def orthogonal_procrustes(A, B):
    M = A.T @ B
    U, S, Vh = np.linalg.svd(M)
    return U @ Vh

def procrustes_project(W_src, target_shape):
    rows_src, cols_src = W_src.shape
    rows_tgt, cols_tgt = target_shape
    
    # 1. SVD for variance extraction
    U, S, Vh = np.linalg.svd(W_src, full_matrices=False)
    
    # 2. Match dimensions
    r = min(len(S), rows_tgt, cols_tgt)
    U_r = U[:rows_tgt, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :cols_tgt]
    
    # 3. Reconstruct and Pad
    W_projected = (U_r * S_r[np.newaxis, :]) @ Vh_r
    final = np.zeros(target_shape, dtype=W_src.dtype)
    final[:W_projected.shape[0], :W_projected.shape[1]] = W_projected
    return final

def redress():
    print("STARTING: Foundation Redress (NumPy Edition)...")
    
    # Load Blueprints
    with open("configs/arch_genesis.yaml", "r") as f:
        genesis_cfg = yaml.safe_load(f)
    with open("configs/arch_sovereign.yaml", "r") as f:
        sovereign_cfg = yaml.safe_load(f)
        
    # Load Genesis Weights
    source_weights = load_file("models/genesis/model.safetensors")
    target_weights = {}
    
    # Group source weights by role for aggregation
    role_groups = {}
    for name, tensor in source_weights.items():
        role = genesis_cfg['layers'][name]['role']
        if role not in role_groups: role_groups[role] = []
        role_groups[role].append(tensor)
        
    # Redress into Sovereign Layers
    for tgt_name, tgt_info in sovereign_cfg['layers'].items():
        role = tgt_info['role']
        tgt_shape = tuple(tgt_info['shape'])
        
        # Aggregate logic: If transitioning 4 layers -> 2 layers, 
        # we fuse the intelligence of two source layers into one target layer
        # For this foundation test, we use the first available intelligence pool
        if role in role_groups and role_groups[role]:
            src_tensor = role_groups[role].pop(0) # Take the first shard
            print(f"  Mapping {role} -> {tgt_name} [Projecting {src_tensor.shape} to {tgt_shape}]")
            
            # Execute Procrustes Projection
            target_weights[tgt_name] = procrustes_project(src_tensor, tgt_shape)
            
    # Save the First New Model
    os.makedirs("models/sovereign", exist_ok=True)
    save_file(target_weights, "models/sovereign/model.safetensors")
    print("\nSUCCESS: REDRESS COMPLETE.")
    print("📁 New Model Created: models/sovereign/model.safetensors")
    print(f"📏 Parameters Transformed: {sum(t.size for t in target_weights.values()):,}")

if __name__ == "__main__":
    redress()
