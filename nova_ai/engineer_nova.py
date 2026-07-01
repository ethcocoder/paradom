import sys
import os
import numpy as np
import ml_dtypes
from safetensors.numpy import load_file, save_file

# 1. Import Paradom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paradom.math.numpy_engine import ParadoxFoundation

def engineer_nova():
    print("STARTING: Nova-AI v1 Construction (English Primary Engine)...")
    
    # Paths
    BASE_DIR = os.path.dirname(__file__)
    SOURCE_PATH = os.path.join(BASE_DIR, "..", "models", "nova_source", "model.safetensors")
    OUTPUT_PATH = os.path.join(BASE_DIR, "model_nova_v1.safetensors")
    
    # 2. Load Instruct Foundation
    print(f"  [CORE] Identifying English Intelligence markers...")
    src_weights = ParadoxFoundation.load_intelligence(SOURCE_PATH)
    nova_weights = {}
    
    # Nova Specs: 12 Layers | 576 Hidden | 1536 FFN (Source Symmetry)
    D_MODEL = 576
    D_FFN = 1536
    
    # 3. Redressing Cycle (Spectral Preservation)
    print("  [SYNC] Projecting Linguistic Coherence into Nova frame...")
    
    # A. Embedding Redress
    emb_src = src_weights["model.embed_tokens.weight"]
    nova_weights["nova.embed.weight"] = ParadoxFoundation.procrustes_project(emb_src, (emb_src.shape[0], D_MODEL))

    # B. 12-Layer Depth Redress (Spectral Blending Strategy)
    print("  [SYNC] Blending Spectral Neighborhoods into Nova frame...")
    
    # We take the top 12 regions and blend 2 neighboring layers for each
    # to 'smooth out' the intelligence transition.
    blends = [
        [0, 1], [2, 3], [5, 6], [9, 10], [13, 14], [15, 16], 
        [17, 18], [21, 22], [23, 24], [25, 26], [27, 28], [29]
    ]
    
    for i, src_group in enumerate(blends):
        # Initialize blended buffer
        blended_src = {}
        
        # Capture Projection names
        proj_names = [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", 
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", 
            "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"
        ]
        
        for p in proj_names:
            tensors = [src_weights[f"model.layers.{idx}.{p}"] for idx in src_group]
            # Average the tensors (Weight Blending)
            blended_src[p] = np.mean(tensors, axis=0)

        # Mapping to Nova
        mapping = {
            f"nova.layer.{i}.q": "self_attn.q_proj.weight",
            f"nova.layer.{i}.k": "self_attn.k_proj.weight",
            f"nova.layer.{i}.v": "self_attn.v_proj.weight",
            f"nova.layer.{i}.o": "self_attn.o_proj.weight",
            f"nova.layer.{i}.g": "mlp.gate_proj.weight",
            f"nova.layer.{i}.u": "mlp.up_proj.weight",
            f"nova.layer.{i}.d": "mlp.down_proj.weight",
            f"nova.layer.{i}.ln1": "input_layernorm.weight",
            f"nova.layer.{i}.ln2": "post_attention_layernorm.weight",
        }
        
        for p_nova, p_src in mapping.items():
            src_tensor = blended_src[p_src]
            if "ln" in p_nova:
                shape = (D_MODEL,)
            elif ".u" in p_nova or ".g" in p_nova:
                shape = (D_FFN, D_MODEL)
            elif ".d" in p_nova:
                shape = (D_MODEL, D_FFN)
            elif ".k" in p_nova or ".v" in p_nova:
                shape = (192, D_MODEL)
            else:
                shape = (D_MODEL, D_MODEL)
            
            nova_weights[p_nova] = ParadoxFoundation.procrustes_project(src_tensor, shape)
            
        if i % 4 == 0: print(f"    Blended Nova Block: {i}/12...")

    # C. Synthesis Redress
    print("    Finalizing Linguistic Synthesis Head...")
    # SmolLM uses tied embeddings, so Nova-Head = Nova-Embed
    nova_weights["nova.head.weight"] = nova_weights["nova.embed.weight"]
    if "model.norm.weight" in src_weights:
        nova_weights["nova.norm.weight"] = ParadoxFoundation.procrustes_project(src_weights["model.norm.weight"], (D_MODEL,))

    # 4. Final Harvest
    save_file(nova_weights, OUTPUT_PATH)
    print(f"\nSUCCESS: Nova-AI v1 is Born. Ready for English Interaction.")

if __name__ == "__main__":
    engineer_nova()
