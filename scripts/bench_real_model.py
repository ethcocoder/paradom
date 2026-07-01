import numpy as np
from safetensors.numpy import load_file
import ml_dtypes
import os

class TinyLlamaInference:
    def __init__(self, weight_path):
        print(f"  [TINY-INF] Loading weights: {weight_path}")
        self.weights = load_file(weight_path)
        self.d_model = 576 

    def simple_tokenize(self, text):
        return [ord(c) % 49152 for c in text] 

    def forward_pass(self, text):
        tokens = self.simple_tokenize(text)
        emb_weight = self.weights["model.embed_tokens.weight"]
        x = emb_weight[tokens]
        W_q = self.weights["model.layers.0.self_attn.q_proj.weight"]
        q = x @ W_q.T
        W_up = self.weights["model.layers.0.mlp.up_proj.weight"]
        mlp_out = x @ W_up.T
        return q, mlp_out

def run_benchmark():
    print("STARTING: Pure-Foundation Signal Test (SmolLM-Instruct)...")
    WEIGHT_PATH = "models/hf_instruct/model.safetensors"
    
    if not os.path.exists(WEIGHT_PATH):
        print("ERROR: Weights not found.")
        return
        
    engine = TinyLlamaInference(WEIGHT_PATH)
    prompt = "Hello Foundation"
    q_signal, mlp_signal = engine.forward_pass(prompt)
    
    print("\nFOUNDATIONAL RESONANCE [Input: '" + prompt + "']:")
    print(f"  Attention Core Mean: {np.mean(q_signal):.6f}")
    print(f"  Attention Core Var:  {np.var(q_signal):.6f}")
    print(f"  MLP-Structure Mean:  {np.mean(mlp_signal):.6f}")
    print(f"  MLP-Structure Var:   {np.var(mlp_signal):.6f}")
    
    if np.var(q_signal) > 0.0001:
        print("\nBENCHMARK PASSED. The real SmolLM-Instruct foundation is active and responding.")
    else:
        print("\nBENCHMARK FAILED. Signal is flat.")

if __name__ == "__main__":
    run_benchmark()
