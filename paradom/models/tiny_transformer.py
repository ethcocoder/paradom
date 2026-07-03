import torch
import torch.nn as nn
from typing import Optional

class TinyTransformer(nn.Module):
    """
    A minimal 2-layer Transformer for Phase 1 Proof of Concept.
    ~10M parameters.
    """
    def __init__(self, vocab_size=50257, d_model=256, n_layers=2, n_heads=8, d_ff=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d_model)
        
        # Explicitly named to match Paradom FunctionalRoleMatcher logic
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.post_attention_layernorm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        h = self.input_layernorm(x)
        
        # Simple Attention
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        
        # (batch, seq, heads, d_head)
        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.d_head).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape)
        
        x = x + self.o_proj(out)
        
        # SwiGLU-like FFN for alignment with Llama
        h = self.post_attention_layernorm(x)
        ffn_out = self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))
        x = x + ffn_out
        
        return x
