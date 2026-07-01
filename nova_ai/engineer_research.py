import sys
import os
import numpy as np
import ml_dtypes
from safetensors.numpy import load_file, save_file
import json

# 1. Import Paradom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paradom.math.numpy_engine import ParadoxFoundation

def engineer_research_nova():
    print("STARTING: Paradom Research Proof (Surgical Weight Equivalence)...")
    
    BASE_DIR = os.path.dirname(__file__)
    SOURCE_PATH = os.path.join(BASE_DIR, "..", "models", "nova_source", "model.safetensors")
    OUTPUT_PATH = os.path.join(BASE_DIR, "model_nova_v1.safetensors")
    
    src_weights = load_file(SOURCE_PATH)
    nova_weights = {}
    
    # Nova Specs (Symmetric to source for maximum proof stability)
    D_MODEL = 576
    D_FFN = 1536
    N_LAYERS = 12
    
    # 2. Winning Ticket Importance Scores (Captured from analyze_importance.py)
    # We use these to weight the Spectral Blending
    energies = {
        0: 0.6621, 1: 0.5407, 2: 0.5428, 3: 0.5002, 4: 0.5088, 5: 0.4965, 
        6: 0.4907, 7: 0.5043, 8: 0.4941, 9: 0.5427, 10: 0.5253, 11: 0.5120, 
        12: 0.5204, 13: 0.5539, 14: 0.6407, 15: 0.5710, 16: 0.5625, 17: 0.5807, 
        18: 0.5689, 19: 0.5350, 20: 0.5523, 21: 0.5788, 22: 0.5701, 23: 0.6077, 
        24: 0.5329, 25: 0.5698, 26: 0.5656, 27: 0.5161, 28: 0.5274, 29: 0.5307
    }

    # 3. Surgical Windows (Overlapping to preserve residual stream)
    # 30 layers mapped to 12. Roughly 2.5 layers per window.
    windows = [
        [0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], 
        [14, 15], [16, 17], [18, 19, 20], [21, 22], [23, 24, 25], 
        [26, 27], [28, 29]
    ]

    # A. Embeddings (Direct Equivalence)
    nova_weights["nova.embed.weight"] = src_weights["model.embed_tokens.weight"]

    # B. Block Redress (Weighted Spectral Blending)
    print("  [SYNC] Performing Weighted Spectral Swaps...")
    
    proj_keys = [
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", 
        "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", 
        "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"
    ]

    for i, window in enumerate(windows):
        # Calculate Weighted average for this window
        window_energies = np.array([energies[idx] for idx in window])
        weights = window_energies / np.sum(window_energies)
        
        for k in proj_keys:
            blended = None
            for idx, w in zip(window, weights):
                t = src_weights[f"model.layers.{idx}.{k}"].astype(np.float32)
                if blended is None:
                    blended = t * w
                else:
                    blended += t * w
            
            # Identify target name
            target_map = {
                "self_attn.q_proj.weight": "q", "self_attn.k_proj.weight": "k",
                "self_attn.v_proj.weight": "v", "self_attn.o_proj.weight": "o",
                "mlp.gate_proj.weight": "g", "mlp.up_proj.weight": "u", "mlp.down_proj.weight": "d",
                "input_layernorm.weight": "ln1", "post_attention_layernorm.weight": "ln2"
            }
            nova_name = f"nova.layer.{i}.{target_map[k]}"
            
            # Projected Swap (Handling GQA if needed)
            shape = blended.shape
            # (Note: In symmetry proof, shapes match exactly, so ParadoxFoundation will direct-copy)
            nova_weights[nova_name] = ParadoxFoundation.procrustes_project(blended, shape)

        if i % 4 == 0: print(f"    Block {i}/12 Redressed...")

    # C. Head & Norm (The Synthesis)
    nova_weights["nova.head.weight"] = nova_weights["nova.embed.weight"]
    nova_weights["nova.norm.weight"] = src_weights["model.norm.weight"]

    save_file(nova_weights, OUTPUT_PATH)
    print("\nSUCCESS: Research Proof Model Constructed.")

if __name__ == "__main__":
    engineer_research_nova()
