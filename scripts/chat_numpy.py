import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer
import json
import os
import ml_dtypes

class TinyLlamaNumpy:
    def __init__(self, model_path, config_path, tokenizer_path):
        print(f"Loading weights from {model_path}...")
        self.raw_weights = load_file(model_path)
        self.weights = {k: v.astype(np.float32) for k, v in self.raw_weights.items()}
        del self.raw_weights 
        
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        self.n_layers = self.config.get("num_hidden_layers", 30)
        self.n_heads = self.config.get("num_attention_heads", 9)
        self.n_kv_heads = self.config.get("num_key_value_heads", 3)
        self.dim = self.config.get("hidden_size", 576)
        self.head_dim = self.dim // self.n_heads
        self.vocab_size = self.config.get("vocab_size", 49152)
        self.norm_eps = self.config.get("rms_norm_eps", 1e-5)
        
        # Precompute RoPE frequencies
        self.freqs_cis = self.precompute_freqs_cis(self.head_dim, 2048)

    def precompute_freqs_cis(self, dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
        t = np.arange(end)
        freqs = np.outer(t, freqs)
        return np.exp(1j * freqs).astype(np.complex64)

    def apply_rotary_emb(self, x, freqs_cis):
        # x: (seq_len, n_heads, head_dim)
        # freqs_cis: (seq_len, 1, head_dim/2)
        x_reshaped = x.reshape(x.shape[0], x.shape[1], -1, 2)
        x_re = x_reshaped[..., 0]
        x_im = x_reshaped[..., 1]
        
        f_re = np.real(freqs_cis)
        f_im = np.imag(freqs_cis)
        
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_re = x_re * f_re - x_im * f_im
        out_im = x_re * f_im + x_im * f_re
        
        out = np.stack([out_re, out_im], axis=-1)
        return out.reshape(*x.shape)

    def rms_norm(self, x, weight):
        pow_x = x**2
        return (x * (1.0 / np.sqrt(pow_x.mean(-1, keepdims=True) + self.norm_eps))) * weight

    def silu(self, x):
        return x * (1.0 / (1.0 + np.exp(-x)))

    def forward(self, tokens):
        h = self.weights["model.embed_tokens.weight"][tokens]
        
        for i in range(self.n_layers):
            # Attention Norm
            norm_x = self.rms_norm(h, self.weights[f"model.layers.{i}.input_layernorm.weight"])
            
            # QKV Projections
            q = norm_x @ self.weights[f"model.layers.{i}.self_attn.q_proj.weight"].T
            k = norm_x @ self.weights[f"model.layers.{i}.self_attn.k_proj.weight"].T
            v = norm_x @ self.weights[f"model.layers.{i}.self_attn.v_proj.weight"].T
            
            # Reshape for multi-head
            q = q.reshape(q.shape[0], self.n_heads, self.head_dim)
            k = k.reshape(k.shape[0], self.n_kv_heads, self.head_dim)
            v = v.reshape(v.shape[0], self.n_kv_heads, self.head_dim)
            
            # RoPE (Simplified for single sequence)
            freqs = self.freqs_cis[:tokens.shape[0]]
            q = self.apply_rotary_emb(q, freqs[:, np.newaxis, :])
            k = self.apply_rotary_emb(k, freqs[:, np.newaxis, :])
            
            # Grouped Query Attention (Simplified - repeats keys/values)
            if self.n_heads != self.n_kv_heads:
                k = np.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
                v = np.repeat(v, self.n_heads // self.n_kv_heads, axis=1)
            
            # scaled dot-product attention
            # q: (t, h, d), k: (s, h, d)
            scores = np.einsum('thd,shd->hds', q, k) / np.sqrt(self.head_dim) # h, d, s is wrong
            # Actually Score = (Q @ K.T) / sqrt(d)
            # Scores: (h, t, s)
            scores = np.einsum('thd,shd->hts', q, k) / np.sqrt(self.head_dim)
            
            # Add causal mask
            seq_len = q.shape[0]
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e10
            scores = scores + mask[np.newaxis, :, :]
            
            probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            
            # Attn Out: (h, t, d)
            attn_out = np.einsum('hts,shd->thd', probs, v) 
            attn_out = attn_out.reshape(seq_len, self.dim)
            
            # Output projection
            h = h + attn_out @ self.weights[f"model.layers.{i}.self_attn.o_proj.weight"].T
            
            # MLP
            norm_x = self.rms_norm(h, self.weights[f"model.layers.{i}.post_attention_layernorm.weight"])
            gate = norm_x @ self.weights[f"model.layers.{i}.mlp.gate_proj.weight"].T
            up = norm_x @ self.weights[f"model.layers.{i}.mlp.up_proj.weight"].T
            ffn_out = (self.silu(gate) * up) @ self.weights[f"model.layers.{i}.mlp.down_proj.weight"].T
            
            h = h + ffn_out
            
        h = self.rms_norm(h, self.weights["model.norm.weight"])
        head_weight = self.weights.get("lm_head.weight", self.weights["model.embed_tokens.weight"])
        logits = h @ head_weight.T
        return logits[-1]

    def generate(self, prompt, max_len=50, temp=0.7):
        tokens = np.array(self.tokenizer.encode(prompt).ids)
        
        for _ in range(max_len):
            logits = self.forward(tokens)
            # Sample
            probs = np.exp(logits / temp)
            probs /= np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
            
            tokens = np.append(tokens, next_token)
            
            if next_token == self.tokenizer.token_to_id("<|endoftext|>"):
                break
                
            yield self.tokenizer.decode([next_token])

def main():
    model_path = "models/nova_source/model.safetensors"
    config_path = "models/nova_source/config.json"
    tokenizer_path = "models/nova_source/tokenizer.json"
    
    bot = TinyLlamaNumpy(model_path, config_path, tokenizer_path)
    
    print("\n[PURE PYTHON CHAT ACTIVE]")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        print("Bot: ", end="", flush=True)
        for chunk in bot.generate(user_input):
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()
