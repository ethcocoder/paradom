import sys
import os
import yaml

# 1. Import the Paradom Framework (New User Perspective)
# We add the root directory to the path to simulate an installed package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paradom.math.numpy_engine import ParadoxFoundation

def build_paradox_model():
    print("[PARADOX AI] - Initiating Foundation Redressing via Paradom Framework...")
    
    # 2. Config & Source paths (Context-Aware)
    BASE_DIR = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(BASE_DIR, "config_ultra.yaml")
    SOURCE_MODEL = os.path.join(BASE_DIR, "..", "models", "hf_source", "model.safetensors")
    OUTPUT_MODEL = os.path.join(BASE_DIR, "model_paradox_ultra.safetensors")
    
    # 3. Use the framework to load intelligence
    print(f"  [PARADOX] Loading source foundation: {SOURCE_MODEL}")
    foundation_weights = ParadoxFoundation.load_intelligence(SOURCE_MODEL)
    
    # 4. Define our new proprietary layers
    # We will redress 2 layers for this Paradox-Ultra prototype
    paradox_weights = {}
    
    # Mapper logic (Internal to Paradox AI)
    # We map LLaMA weights from SmolLM into Paradox Ultra slots
    mapping_plan = {
        "paradox.ultra.layer_0.att_q": "model.layers.0.self_attn.q_proj.weight",
        "paradox.ultra.layer_0.att_k": "model.layers.0.self_attn.k_proj.weight",
        "paradox.ultra.layer_0.att_v": "model.layers.0.self_attn.v_proj.weight",
        "paradox.ultra.layer_1.att_q": "model.layers.1.self_attn.q_proj.weight",
        "paradox.ultra.layer_1.att_k": "model.layers.1.self_attn.k_proj.weight",
        "paradox.ultra.layer_1.att_v": "model.layers.1.self_attn.v_proj.weight"
    }
    
    print("  [PARADOX] Executing Procrustes Projection...")
    for target_name, source_name in mapping_plan.items():
        src_tensor = foundation_weights[source_name]
        
        # We project into the Paradox Ultra dimension: 1024x1024
        target_shape = (1024, 1024)
        
        print(f"    Mapping {source_name} --> {target_name} ({target_shape})")
        paradox_weights[target_name] = ParadoxFoundation.procrustes_project(src_tensor, target_shape)
        
    # 5. Persist the Paradox AI model
    ParadoxFoundation.save_redressed(paradox_weights, OUTPUT_MODEL)
    print("\nSUCCESS: [PARADOX AI] COMPLETE. New Foundation: {OUTPUT_MODEL}")
    print(f"📏 Paradox Dimension: 1024 | Layers: 2 Prototype | Status: Sovereign")

if __name__ == "__main__":
    build_paradox_model()
