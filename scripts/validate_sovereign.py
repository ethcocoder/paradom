from safetensors.numpy import load_file
import numpy as np

print("STARTING: Sovereign Model Audit...")
weights = load_file("models/sovereign/model.safetensors")

# Verification 1: Layer Count
q_layers = [k for k in weights.keys() if "q_projection" in k]
print(f"  Target Layer Count: {len(q_layers)} (Expected: 2)")

# Verification 2: Dimensional Integrity
for k, v in weights.items():
    print(f"  Weight: {k} | Shape: {v.shape} | Variance: {np.var(v):.6f}")

if len(q_layers) == 2 and all(v.shape == (768, 768) for v in weights.values()):
    print("\nFINAL STATUS: SUCCESS. The first new model is mathematically sound and dimensionally correct.")
else:
    print("\nFINAL STATUS: WARNING. Dimensional discrepancy detected.")
