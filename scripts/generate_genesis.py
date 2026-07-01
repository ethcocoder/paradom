import numpy as np
from safetensors.numpy import save_file
import os

# Create directory
os.makedirs("models/genesis", exist_ok=True)

# Generate 4 layers of weights (512x512)
weights = {}
for i in range(4):
    weights[f"layers.{i}.attention.wq.weight"] = np.random.randn(512, 512).astype(np.float32)
    weights[f"layers.{i}.attention.wk.weight"] = np.random.randn(512, 512).astype(np.float32)
    weights[f"layers.{i}.attention.wv.weight"] = np.random.randn(512, 512).astype(np.float32)

save_file(weights, "models/genesis/model.safetensors")
print("SUCCESS: Genesis model generated with 4 layers (512x512).")
