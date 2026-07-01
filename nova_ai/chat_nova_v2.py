"""
Nova-AI v2: Pure Direct-Copy Redress
Uses the EXACT same weight format as the source model (model.layers.X.*) 
but only copies 12 selected layers. This isolates whether the problem is in 
the engineering (copy) step or the inference (forward pass) step.
"""
import numpy as np
from safetensors.numpy import load_file, save_file
from tokenizers import Tokenizer
import json
import os
import ml_dtypes

class NovaV2Inference:
    def __init__(self, model_path, config_path, tokenizer_path):
        print(f"Loading weights from {model_path}...")
        raw = load_file(model_path)
        self.weights = {k: v.astype(np.float32) for k, v in raw.items()}
        del raw
        
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Use source config exactly
        self.n_layers = self.config.get("num_hidden_layers", 30)
        self.n_heads = self.config.get("num_attention_heads", 9)
        self.n_kv_heads = self.config.get("num_key_value_heads", 3)
        self.dim = self.config.get("hidden_size", 576)
        self.head_dim = self.dim // self.n_heads
        self.norm_eps = self.config.get("rms_norm_eps", 1e-5)
        
        self.freqs_cis = self.precompute_freqs_cis(self.head_dim, 2048)

    def precompute_freqs_cis(self, dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
        t = np.arange(end)
        freqs = np.outer(t, freqs)
        return np.exp(1j * freqs).astype(np.complex64)

    def apply_rotary_emb(self, x, freqs_cis):
        x_reshaped = x.reshape(x.shape[0], x.shape[1], -1, 2)
        x_re, x_im = x_reshaped[..., 0], x_reshaped[..., 1]
        f_re, f_im = np.real(freqs_cis), np.imag(freqs_cis)
        out_re = x_re * f_re - x_im * f_im
        out_im = x_re * f_im + x_im * f_re
        return np.stack([out_re, out_im], axis=-1).reshape(*x.shape)

    def rms_norm(self, x, weight):
        return (x * (1.0 / np.sqrt(np.mean(x**2, -1, keepdims=True) + self.norm_eps))) * weight

    def silu(self, x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))

    def forward(self, tokens):
        h = self.weights["model.embed_tokens.weight"][tokens]
        
        for i in range(self.n_layers):
            norm_x = self.rms_norm(h, self.weights[f"model.layers.{i}.input_layernorm.weight"])
            
            q = norm_x @ self.weights[f"model.layers.{i}.self_attn.q_proj.weight"].T
            k = norm_x @ self.weights[f"model.layers.{i}.self_attn.k_proj.weight"].T
            v = norm_x @ self.weights[f"model.layers.{i}.self_attn.v_proj.weight"].T
            
            q = q.reshape(q.shape[0], self.n_heads, self.head_dim)
            k = k.reshape(k.shape[0], self.n_kv_heads, self.head_dim)
            v = v.reshape(v.shape[0], self.n_kv_heads, self.head_dim)
            
            freqs = self.freqs_cis[:tokens.shape[0]]
            q = self.apply_rotary_emb(q, freqs[:, np.newaxis, :])
            k = self.apply_rotary_emb(k, freqs[:, np.newaxis, :])
            
            if self.n_heads != self.n_kv_heads:
                k = np.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
                v = np.repeat(v, self.n_heads // self.n_kv_heads, axis=1)
            
            scores = np.einsum('thd,shd->hts', q, k) / np.sqrt(self.head_dim)
            seq_len = q.shape[0]
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e10
            scores += mask[np.newaxis, :, :]
            
            probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            
            attn_out = np.einsum('hts,shd->thd', probs, v).reshape(seq_len, self.dim)
            h = h + attn_out @ self.weights[f"model.layers.{i}.self_attn.o_proj.weight"].T
            
            norm_x = self.rms_norm(h, self.weights[f"model.layers.{i}.post_attention_layernorm.weight"])
            gate = norm_x @ self.weights[f"model.layers.{i}.mlp.gate_proj.weight"].T
            up = norm_x @ self.weights[f"model.layers.{i}.mlp.up_proj.weight"].T
            ffn_out = (self.silu(gate) * up) @ self.weights[f"model.layers.{i}.mlp.down_proj.weight"].T
            h = h + ffn_out
            
        h = self.rms_norm(h, self.weights["model.norm.weight"])
        head_weight = self.weights.get("lm_head.weight", self.weights["model.embed_tokens.weight"])
        logits = h @ head_weight.T
        return logits[-1]

    def generate(self, prompt, max_len=50):
        tokens = np.array(self.tokenizer.encode(prompt).ids)
        for _ in range(max_len):
            logits = self.forward(tokens)
            
            # Greedy search
            next_token = int(np.argmax(logits))
            
            if next_token in tokens[-10:]: # Simple local repetition break
                # If it's repeating, pick the second best
                logits[next_token] = -float('Inf')
                next_token = int(np.argmax(logits))

            tokens = np.append(tokens, next_token)
            if next_token in [1, 2]: break
            yield self.tokenizer.decode([next_token])


def engineer_nova_v2():
    """Creates Nova by directly copying 12 selected layers from the source,
    using the same key naming as the source model. Zero projection."""
    
    print("ENGINEERING: Nova-AI v2 (Direct-Copy, 12 Layers)...")
    src = load_file("models/nova_source/model.safetensors")
    nova = {}
    
    # Select 12 layers covering early, mid, and late regions
    selected = [0, 1, 2, 5, 9, 13, 17, 21, 24, 27, 28, 29]
    
    # Copy embedding
    nova["model.embed_tokens.weight"] = src["model.embed_tokens.weight"]
    
    # Copy selected layers, renumbering them 0-11
    for new_idx, old_idx in enumerate(selected):
        for key in src:
            if f"model.layers.{old_idx}." in key:
                new_key = key.replace(f"model.layers.{old_idx}.", f"model.layers.{new_idx}.")
                nova[new_key] = src[key]
    
    # Copy final norm
    nova["model.norm.weight"] = src["model.norm.weight"]
    
    # Save
    save_file(nova, "nova_ai/model_nova_v2.safetensors")
    
    # Write config
    config = json.load(open("models/nova_source/config.json"))
    config["num_hidden_layers"] = 12
    json.dump(config, open("nova_ai/config_nova_v2.json", "w"), indent=2)
    
    print(f"SUCCESS: Nova-AI v2 (12 layers from {selected})")
    return "nova_ai/model_nova_v2.safetensors", "nova_ai/config_nova_v2.json"


def main():
    # Step 1: Engineer
    model_path, config_path = engineer_nova_v2()
    tokenizer_path = "models/nova_source/tokenizer.json"
    
    # Step 2: Chat
    nova = NovaV2Inference(model_path, config_path, tokenizer_path)
    print(f"\n[NOVA-AI v2: DIRECT-COPY DEPTH REDRESS]")
    print(f"12 layers surgically extracted from 30. Zero projection noise.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        # SmolLM Chat Template
        full_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        print("Nova: ", end="", flush=True)
        for chunk in nova.generate(full_prompt): 
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()
