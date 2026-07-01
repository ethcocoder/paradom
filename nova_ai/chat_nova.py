import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer
import json
import os
import ml_dtypes

class NovaInference:
    def __init__(self, model_path, tokenizer_path):
        print(f"Loading Nova-AI weights from {model_path}...")
        raw = load_file(model_path)
        self.weights = {k: v.astype(np.float32) for k, v in raw.items()}
        del raw
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        self.n_layers = 12
        self.dim = 576
        self.n_heads = 9
        self.n_kv_heads = 3
        self.head_dim = self.dim // self.n_heads
        self.norm_eps = 1e-5
        
        # Precompute RoPE
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
        return x * (1.0 / (1.0 + np.exp(-x)))

    def forward(self, tokens):
        h = self.weights["nova.embed.weight"][tokens]
        
        for i in range(self.n_layers):
            # Attention
            norm_x = self.rms_norm(h, self.weights[f"nova.layer.{i}.ln1"])
            
            q = norm_x @ self.weights[f"nova.layer.{i}.q"].T
            k = norm_x @ self.weights[f"nova.layer.{i}.k"].T
            v = norm_x @ self.weights[f"nova.layer.{i}.v"].T
            
            q = q.reshape(q.shape[0], self.n_heads, self.head_dim)
            k = k.reshape(k.shape[0], self.n_kv_heads, self.head_dim)
            v = v.reshape(v.shape[0], self.n_kv_heads, self.head_dim)
            
            freqs = self.freqs_cis[:tokens.shape[0]]
            q = self.apply_rotary_emb(q, freqs[:, np.newaxis, :])
            k = self.apply_rotary_emb(k, freqs[:, np.newaxis, :])
            
            # GQA Repeat
            k = np.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
            v = np.repeat(v, self.n_heads // self.n_kv_heads, axis=1)
            
            scores = np.einsum('thd,shd->hts', q, k) / np.sqrt(self.head_dim)
            mask = np.triu(np.ones((q.shape[0], q.shape[0])), k=1) * -1e10
            scores += mask[np.newaxis, :, :]
            
            probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            
            # Attn Out: (h, t, d)
            attn_out = np.einsum('hts,shd->thd', probs, v).reshape(tokens.shape[0], self.dim)
            h = h + attn_out @ self.weights[f"nova.layer.{i}.o"].T
            
            # MLP
            norm_x = self.rms_norm(h, self.weights[f"nova.layer.{i}.ln2"])
            g = norm_x @ self.weights[f"nova.layer.{i}.g"].T
            u = norm_x @ self.weights[f"nova.layer.{i}.u"].T
            ffn_out = (self.silu(g) * u) @ self.weights[f"nova.layer.{i}.d"].T
            h = h + ffn_out
            
        h = self.rms_norm(h, self.weights["nova.norm.weight"])
        logits = h @ self.weights["nova.head.weight"].T
        return logits[-1]

    def generate(self, prompt, max_len=50, temp=0.7, top_k=40, repetition_penalty=1.2):
        tokens = np.array(self.tokenizer.encode(prompt).ids)
        for _ in range(max_len):
            logits = self.forward(tokens)
            
            # Repetition Penalty
            for token_id in set(tokens):
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty
            
            # Top-K filtering
            indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
            logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = np.exp(logits / temp)
            probs /= np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
            
            tokens = np.append(tokens, next_token)
            if next_token == self.tokenizer.token_id_to_id("<|endoftext|>") if hasattr(self.tokenizer, "token_id_to_id") else next_token in [2, 1]: break
            yield self.tokenizer.decode([next_token])

def main():
    model_path = "nova_ai/model_nova_v1.safetensors"
    tokenizer_path = "models/nova_source/tokenizer.json"
    
    nova = NovaInference(model_path, tokenizer_path)
    print("\n[NOVA-AI v1: TRANSLATED INTELLIGENCE ACTIVE]")
    print("This model has never been trained. It was 'redressed' from SmolLM.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        # SmolLM Chat Template
        full_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        print("Nova: ", end="", flush=True)
        for chunk in nova.generate(full_prompt, temp=0.2): # Consistency
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()
