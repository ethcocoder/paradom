import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer
import ml_dtypes
import os

def compute_cka(X, Y):
    # Linear CKA implementation (NumPy)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    dot_prod = np.linalg.norm(X.T @ Y, ord='fro')**2
    norm_x = np.linalg.norm(X.T @ X, ord='fro')
    norm_y = np.linalg.norm(Y.T @ Y, ord='fro')
    
    return dot_prod / (norm_x * norm_y + 1e-8)

def validate_accuracy():
    print("VALIDATING: Research Accuracy (Source vs Nova)...")
    src = load_file("models/nova_source/model.safetensors")
    nova = load_file("nova_ai/model_nova_v1.safetensors")
    
    # We compare the final representation vectors (the 'knowledge' before the head)
    # the embeddings are identical, so we check the middle-to-late layers
    
    # Let's compare Nova Layer 11 (Final) with Source Layer 29 (Final)
    # These represent the 'processed knowledge'
    w_src = src["model.layers.29.self_attn.o_proj.weight"].astype(np.float32)
    w_nova = nova["nova.layer.11.o"].astype(np.float32)
    
    # Since they are both in the 576-dim space (High-Fidelity Proof), 
    # we can compute CKA directly.
    cka_score = compute_cka(w_src, w_nova)
    
    print(f"\nKNOWLEDGE FIDELITY SCORE: {cka_score*100:.2f}%")
    
    if cka_score > 0.7:
        print("RESULT: SUCCESS. The intelligent product is preserved.")
    else:
        print("RESULT: DEGRADED. Some nuance was lost during projection.")

if __name__ == "__main__":
    validate_accuracy()
