import numpy as np
from safetensors.numpy import load_file
import ml_dtypes
import json

def analyze_sovereign_importance():
    print("ANALYZING: Source Model Spectrum (Winning Ticket Search)...")
    weights = load_file("models/nova_source/model.safetensors")
    
    layer_scores = []
    
    for i in range(30):
        # We analyze the 'energy' across all projections in the block
        # Q, K, V, O, Up, Down, Gate
        projections = [
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight"
        ]
        
        block_energy = 0
        for p in projections:
            W = weights[p].astype(np.float32)
            S = np.linalg.svd(W, compute_uv=False)
            # Energy = sum of top 20% singular values squared
            top_k = int(len(S) * 0.2) + 1
            energy = np.sum(S[:top_k]**2) / np.sum(S**2)
            block_energy += energy
            
        avg_energy = block_energy / len(projections)
        layer_scores.append((i, avg_energy))
        
    # Sort by importance
    print("\nALL LAYER SPECTRAL ENERGIES:")
    for idx, score in layer_scores:
        print(f"  Layer {idx:2d}: {score:.4f}")
        
    sorted_layers = sorted(layer_scores, key=lambda x: x[1], reverse=True)
    
    print("\nTOP 12 WINNING TICKETS (Highest Spectral Energy):")
    top_12 = [idx for idx, score in sorted_layers[:12]]
    top_12.sort()
    for idx in top_12:
        score = next(s for i, s in layer_scores if i == idx)
        print(f"  Layer {idx:2d}: Energy {score:.4f}")
        
    return top_12

if __name__ == "__main__":
    analyze_sovereign_importance()
