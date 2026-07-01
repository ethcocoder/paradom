import sys
import os
import numpy as np
import ml_dtypes
from safetensors.numpy import load_file, save_file
import json

# Import Paradom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paradom.math.numpy_engine import ParadoxFoundation

def engineer_supersymmetry_nova():
    print("STARTING: Super-Symmetric Proof (Dimensional Redress)...")
    
    BASE_DIR = os.path.dirname(__file__)
    SOURCE_PATH = os.path.join(BASE_DIR, "..", "models", "nova_source", "model.safetensors")
    OUTPUT_PATH = os.path.join(BASE_DIR, "model_nova_v1.safetensors")
    
    src_weights = load_file(SOURCE_PATH)
    nova_weights = {}
    
    # Target Architecture: 30 Layers | 512 Hidden | 8 Heads | 2048 FFN
    D_TGT = 512
    D_FFN_TGT = 2048
    N_LAYERS = 30
    
    # A. Embeddings
    emb_src = src_weights["model.embed_tokens.weight"]
    nova_weights["nova.embed.weight"] = ParadoxFoundation.procrustes_project(emb_src, (emb_src.shape[0], D_TGT))

    # B. 30-Layer Surgical Projection
    print("  [SYNC] Projecting 30 Layers into 512-dim frame...")
    
    for i in range(N_LAYERS):
        mapping = {
            f"nova.layer.{i}.q": (f"model.layers.{i}.self_attn.q_proj.weight", (D_TGT, D_TGT)),
            f"nova.layer.{i}.k": (f"model.layers.{i}.self_attn.k_proj.weight", (D_TGT, D_TGT)), # MHA conversion (512, 512)
            f"nova.layer.{i}.v": (f"model.layers.{i}.self_attn.v_proj.weight", (D_TGT, D_TGT)), # MHA conversion (512, 512)
            f"nova.layer.{i}.o": (f"model.layers.{i}.self_attn.o_proj.weight", (D_TGT, D_TGT)),
            f"nova.layer.{i}.g": (f"model.layers.{i}.mlp.gate_proj.weight", (D_FFN_TGT, D_TGT)),
            f"nova.layer.{i}.u": (f"model.layers.{i}.mlp.up_proj.weight", (D_FFN_TGT, D_TGT)),
            f"nova.layer.{i}.d": (f"model.layers.{i}.mlp.down_proj.weight", (D_TGT, D_FFN_TGT)),
            f"nova.layer.{i}.ln1": (f"model.layers.{i}.input_layernorm.weight", (D_TGT,)),
            f"nova.layer.{i}.ln2": (f"model.layers.{i}.post_attention_layernorm.weight", (D_TGT,)),
        }
        
        for k_nova, (k_src, shape) in mapping.items():
            src_tensor = src_weights[k_src]
            nova_weights[k_nova] = ParadoxFoundation.procrustes_project(src_tensor, shape)

        if i % 10 == 0: print(f"    Block {i}/30 Redressed...")

    # C. Head & Norm
    nova_weights["nova.head.weight"] = nova_weights["nova.embed.weight"]
    nova_weights["nova.norm.weight"] = ParadoxFoundation.procrustes_project(src_weights["model.norm.weight"], (D_TGT,))

    save_file(nova_weights, OUTPUT_PATH)
    print("\nSUCCESS: Super-Symmetric Model Born.")

if __name__ == "__main__":
    engineer_supersymmetry_nova()
