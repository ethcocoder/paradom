import sys
import os
import numpy as np
import ml_dtypes
from safetensors.numpy import load_file, save_file
import json

# Import Paradom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paradom.math.numpy_engine import ParadoxFoundation

def engineer_symmetry_nova():
    print("STARTING: Winning Ticket Symmetry Proof...")
    
    BASE_DIR = os.path.dirname(__file__)
    SOURCE_PATH = os.path.join(BASE_DIR, "..", "models", "nova_source", "model.safetensors")
    OUTPUT_PATH = os.path.join(BASE_DIR, "model_nova_v1.safetensors")
    
    src_weights = load_file(SOURCE_PATH)
    nova_weights = {}
    
    # 12 Highest-Energy Spectral Layers (The Winning Tickets)
    selected_layers = [0, 13, 14, 15, 16, 17, 18, 21, 22, 23, 25, 26]
    
    # A. Embeddings (Identity)
    nova_weights["nova.embed.weight"] = src_weights["model.embed_tokens.weight"]

    # B. Block Redress (Direct Surgical Swap)
    print("  [SYNC] Swapping 12 Winning Tickets...")
    
    for i, src_idx in enumerate(selected_layers):
        mapping = {
            f"nova.layer.{i}.q": f"model.layers.{src_idx}.self_attn.q_proj.weight",
            f"nova.layer.{i}.k": f"model.layers.{src_idx}.self_attn.k_proj.weight",
            f"nova.layer.{i}.v": f"model.layers.{src_idx}.self_attn.v_proj.weight",
            f"nova.layer.{i}.o": f"model.layers.{src_idx}.self_attn.o_proj.weight",
            f"nova.layer.{i}.g": f"model.layers.{src_idx}.mlp.gate_proj.weight",
            f"nova.layer.{i}.u": f"model.layers.{src_idx}.mlp.up_proj.weight",
            f"nova.layer.{i}.d": f"model.layers.{src_idx}.mlp.down_proj.weight",
            f"nova.layer.{i}.ln1": f"model.layers.{src_idx}.input_layernorm.weight",
            f"nova.layer.{i}.ln2": f"model.layers.{src_idx}.post_attention_layernorm.weight",
        }
        
        for k_nova, k_src in mapping.items():
            nova_weights[k_nova] = src_weights[k_src] # Direct copy

        if i % 4 == 0: print(f"    Symmetric Block {i}/12 Redressed...")

    # C. Head & Norm
    nova_weights["nova.head.weight"] = nova_weights["nova.embed.weight"]
    nova_weights["nova.norm.weight"] = src_weights["model.norm.weight"]

    save_file(nova_weights, OUTPUT_PATH)
    print("\nSUCCESS: Symmetry Proof Constructed.")

if __name__ == "__main__":
    engineer_symmetry_nova()
