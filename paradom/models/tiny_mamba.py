import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TinyMamba(nn.Module):
    """
    A minimal 2-layer Mamba for Phase 1 Proof of Concept.
    Pure-PyTorch implementation (no CUDA kernels required).
    """
    def __init__(self, vocab_size=50257, d_model=256, n_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, d_model // 16)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d  = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv-1)
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x):
        # Simplified Mamba forward pass
        # x: (b, s, d)
        residual = x
        x = self.norm(x)
        
        # in_proj splits into x and z (gate)
        xz = self.in_proj(x) # (b, s, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv
        x = x.transpose(1, 2) # (b, d_inner, s)
        x = self.conv1d(x)[:, :, :residual.shape[1]]
        x = x.transpose(1, 2) # (b, s, d_inner)
        x = F.silu(x)
        
        # SSM (Simple Scan)
        # Note: This is an extremely simplified version for Phase 1
        # In a real Mamba, this would be a parallel scan.
        # But for POC weight equivalence, we focus on the projection weights.
        y = self._ssm(x)
        
        # Gating
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        return out + residual

    def _ssm(self, x):
        """
        Discretized SSM scan:
        h_t = exp(ΔA)h_{t-1} + ΔBx_t
        y_t = Ch_t + Dx_t
        """
        b, s, d = x.shape
        d_state = self.A_log.shape[1]
        
        # Projections
        x_dbl = self.x_proj(x) # (b, s, dt_rank + 2*d_state)
        dt_raw, B, C = torch.split(x_dbl, [self.dt_rank, d_state, d_state], dim=-1)
        
        # Discretize Δ
        dt = F.softplus(self.dt_proj(dt_raw)) # (b, s, d_inner)
        
        # Discretize A (ΔA)
        # self.A_log is (d_inner, d_state)
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # Scan (Sequential for simplicity in Phase 1 POC)
        # Note: In production this is a parallel prefix scan.
        h = torch.zeros(b, d, d_state, device=x.device, dtype=x.dtype)
        y = torch.zeros(b, s, d, device=x.device, dtype=x.dtype)
        
        for t in range(s):
            # Δ: (b, d_inner)
            dt_t = dt[:, t, :].unsqueeze(-1) # (b, d, 1)
            
            # bar_A = exp(ΔA) -> (b, d, d_state)
            bar_A = torch.exp(dt_t * A)
            
            # bar_B = ΔB -> (b, d, d_state)
            # B: (b, d_state) -> (b, 1, d_state)
            B_t = B[:, t, :].unsqueeze(1)
            bar_B = dt_t * B_t
            
            # x_t: (b, d) -> (b, d, 1)
            x_t = x[:, t, :].unsqueeze(-1)
            
            # h_t = bar_A * h_{t-1} + bar_B * x_t
            h = bar_A * h + bar_B * x_t
            
            # y_t = C_t * h_t + D * x_t
            # C: (b, d_state) -> (b, 1, d_state)
            C_t = C[:, t, :].unsqueeze(1)
            y[:, t, :] = (C_t @ h.transpose(1, 2)).squeeze(1) + (self.D * x[:, t, :])
            
        return y
