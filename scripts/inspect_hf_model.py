from safetensors.numpy import load_file
import numpy as np

print("STARTING: Real-World Model Inspection (SmolLM-135M) - FINAL...")
weights = load_file("models/nova_source/model.safetensors")

# Count total layers
layer_indices = set()
for name in weights.keys():
    if "layers" in name:
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                layer_indices.add(int(p))

layer_count = max(layer_indices) + 1 if layer_indices else 0
param_count = sum(t.size for t in weights.values())

print(f"FOUNDATION REPORT:")
print(f"  Architecture: LLaMA-style (GQA)")
print(f"  Total Layers: {layer_count}")
print(f"  Hidden Dimension: 576")
print(f"  Total Parameters: {param_count:,}")

# Print a few more layer names for role matching
print("\nKEY LAYER NAMES:")
for name in list(weights.keys())[:50]:
    if "layers.0" in name:
        print(f"  {name}")
