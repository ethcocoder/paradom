"""
Cross-Architecture Mapper: LLaMA → GPT-2
==========================================
Maps weights from a LLaMA-style model (SmolLM) to GPT-2.

Key transformations:
1. Separate Q/K/V → Fused QKV (c_attn)
2. SwiGLU FFN → GELU FFN
3. nn.Linear layout → Conv1D layout (transpose)
4. RMSNorm → LayerNorm (copy gamma, create beta=zeros)
5. RoPE → Learned position embeddings (random init)
6. Layer selection: 30 → 12 (last 12 layers)
7. Vocab padding: 49152 → 50257
8. d_model projection: 576 → 768 (PCA or zero-pad)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from paradom.core.enums import SwapType, FunctionalRole
from paradom.core.types import EquivalencePair, EquivalenceMap, WeightProduct
from paradom.core.cka import weight_cka


class LlamaToGPT2Mapper:
    """
    Maps LLaMA-style weights to GPT-2 architecture.

    Handles all architectural differences:
    - Attention: separate Q/K/V → fused QKV
    - FFN: SwiGLU (3 matrices) → GELU (2 matrices)
    - Norm: RMSNorm → LayerNorm
    - Position: RoPE → learned embeddings
    - Weight layout: nn.Linear → Conv1D (transpose)
    """

    def __init__(self):
        self.d_model_src = None
        self.d_model_tgt = 768
        self.d_inner_src = None
        self.d_inner_tgt = 3072
        self.n_heads_src = None
        self.n_heads_tgt = 12
        self.head_dim = 64
        self.n_layers_src = 30
        self.n_layers_tgt = 12
        self.vocab_src = 49152
        self.vocab_tgt = 50257
        self.selected_layers = None  # Last 12 layers
        self.P_dmodel = None  # Projection matrix for d_model

    def convert(
        self,
        source_products: List[WeightProduct],
        target_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Tensor], EquivalenceMap]:
        """
        Convert LLaMA weights to GPT-2 state_dict.
        """
        # Extract source config from weights
        self._extract_source_config(source_products)

        # Select last 12 layers
        all_layers = sorted(set(wp.layer_index for wp in source_products if wp.layer_index >= 0))
        self.selected_layers = all_layers[-self.n_layers_tgt:]
        print(f"  [Mapper] Selected layers {self.selected_layers[0]}-{self.selected_layers[-1]} from {len(all_layers)} total")

        # Build projection matrix for d_model
        self._build_projection(source_products)

        # Group source products by layer
        layers: Dict[int, Dict[FunctionalRole, WeightProduct]] = {}
        global_params: Dict[FunctionalRole, WeightProduct] = {}

        for wp in source_products:
            if wp.layer_index >= 0:
                if wp.layer_index not in layers:
                    layers[wp.layer_index] = {}
                layers[wp.layer_index][wp.functional_role] = wp
            else:
                global_params[wp.functional_role] = wp

        target = {}
        pairs = []
        cka_scores = {}

        # 1. Embedding
        if FunctionalRole.EMBEDDING in global_params:
            wp = global_params[FunctionalRole.EMBEDDING]
            wte = self._project_embedding(wp.tensor.float())
            target["transformer.wte.weight"] = wte
            pairs.append(EquivalencePair(wp, "transformer.wte.weight", wte.shape,
                         cka_score=weight_cka(wp.tensor, wte[:wp.tensor.shape[0]]),
                         swap_type=SwapType.PROJECTED, confidence=0.8))

        # 2. Position embeddings (random init)
        wpe = torch.randn(self.vocab_tgt, self.d_model_tgt) * 0.02
        # Register as a dummy weight for the state dict
        target["transformer.wpe.weight"] = wpe

        # 3. Final layer norm
        if FunctionalRole.FINAL_NORMALIZATION in global_params:
            wp = global_params[FunctionalRole.FINAL_NORMALIZATION]
            ln_f = self._convert_norm(wp.tensor.float())
            target["transformer.ln_f.weight"] = ln_f
            target["transformer.ln_f.bias"] = torch.zeros(self.d_model_tgt)

        # 4. Transformer layers
        for tgt_idx, src_idx in enumerate(self.selected_layers):
            if src_idx not in layers:
                continue
            l_src = layers[src_idx]

            # Pre-attention norm
            if FunctionalRole.NORMALIZATION in l_src:
                wp = l_src[FunctionalRole.NORMALIZATION]
                ln = self._convert_norm(wp.tensor.float())
                target[f"transformer.h.{tgt_idx}.ln_1.weight"] = ln
                target[f"transformer.h.{tgt_idx}.ln_1.bias"] = torch.zeros(self.d_model_tgt)

            # QKV: separate Q/K/V → fused c_attn
            q_w, k_w, v_w = self._extract_qkv(l_src)
            fused_qkv = self._fuse_qkv(q_w, k_w, v_w)
            # Conv1D layout: transpose to (in, out)
            target[f"transformer.h.{tgt_idx}.attn.c_attn.weight"] = fused_qkv.T
            target[f"transformer.h.{tgt_idx}.attn.c_attn.bias"] = torch.zeros(3 * self.d_model_tgt)

            # Attention output
            if FunctionalRole.CONTEXT_OUTPUT in l_src:
                wp = l_src[FunctionalRole.CONTEXT_OUTPUT]
                o_proj = self._project_dmodel(wp.tensor.float())
                target[f"transformer.h.{tgt_idx}.attn.c_proj.weight"] = o_proj.T
                target[f"transformer.h.{tgt_idx}.attn.c_proj.bias"] = torch.zeros(self.d_model_tgt)

            # Post-attention norm
            if FunctionalRole.POST_NORMALIZATION in l_src:
                wp = l_src[FunctionalRole.POST_NORMALIZATION]
                ln = self._convert_norm(wp.tensor.float())
                target[f"transformer.h.{tgt_idx}.ln_2.weight"] = ln
                target[f"transformer.h.{tgt_idx}.ln_2.bias"] = torch.zeros(self.d_model_tgt)

            # FFN: SwiGLU → GELU
            if FunctionalRole.FFN_GATE in l_src and FunctionalRole.FFN_EXPAND in l_src:
                wp_gate = l_src[FunctionalRole.FFN_GATE]
                wp_up = l_src[FunctionalRole.FFN_EXPAND]
                c_fc = self._convert_ffn_up(wp_gate.tensor.float(), wp_up.tensor.float())
                target[f"transformer.h.{tgt_idx}.mlp.c_fc.weight"] = c_fc.T
                target[f"transformer.h.{tgt_idx}.mlp.c_fc.bias"] = torch.zeros(self.d_inner_tgt)

            if FunctionalRole.FFN_CONTRACT in l_src:
                wp = l_src[FunctionalRole.FFN_CONTRACT]
                c_proj = self._convert_ffn_down(wp.tensor.float())
                target[f"transformer.h.{tgt_idx}.mlp.c_proj.weight"] = c_proj.T
                target[f"transformer.h.{tgt_idx}.mlp.c_proj.bias"] = torch.zeros(self.d_model_tgt)

        mean_cka = sum(cka_scores.values()) / max(len(cka_scores), 1)
        return target, EquivalenceMap(
            source_model="SmolLM-135M",
            target_architecture="GPT-2-Small",
            pairs=pairs,
            unmapped_source=[],
            uninitialized_target=[],
            mean_cka=mean_cka,
            estimated_quality_tier="cross_architecture"
        )

    def _extract_source_config(self, source_products):
        """Extract dimensions from source weights."""
        for wp in source_products:
            if wp.functional_role == FunctionalRole.EMBEDDING:
                self.vocab_src = wp.tensor.shape[0]
                self.d_model_src = wp.tensor.shape[1]
                break

        for wp in source_products:
            if wp.functional_role == FunctionalRole.FFN_GATE:
                self.d_inner_src = wp.tensor.shape[0]
                break

        # Infer head count from Q projection
        for wp in source_products:
            if wp.functional_role == FunctionalRole.CONTEXT_QUERY:
                self.n_heads_src = wp.tensor.shape[0] // self.head_dim
                break

        print(f"  [Mapper] Source: d_model={self.d_model_src}, d_inner={self.d_inner_src}, "
              f"heads={self.n_heads_src}, layers={self.n_layers_src}, vocab={self.vocab_src}")
        print(f"  [Mapper] Target: d_model={self.d_model_tgt}, d_inner={self.d_inner_tgt}, "
              f"heads={self.n_heads_tgt}, layers={self.n_layers_tgt}, vocab={self.vocab_tgt}")

    def _build_projection(self, source_products):
        """Build PCA projection matrix from embedding for d_model 576→768."""
        for wp in source_products:
            if wp.functional_role == FunctionalRole.EMBEDDING:
                W = wp.tensor.float()  # (vocab, 576)
                # PCA on embedding
                W_centered = W - W.mean(dim=0)
                cov = (W_centered.T @ W_centered) / (W_centered.shape[0] - 1)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                # Sort descending
                sorted_idx = torch.argsort(eigenvalues, descending=True)
                eigenvectors = eigenvectors[:, sorted_idx]
                # Take top 576 eigenvectors, pad to 768
                P = torch.zeros(self.d_model_src, self.d_model_tgt)
                P[:, :self.d_model_src] = eigenvectors
                self.P_dmodel = P
                print(f"  [Mapper] Built d_model projection: {self.d_model_src} → {self.d_model_tgt}")
                break

    def _project_dmodel(self, W: Tensor) -> Tensor:
        """Project d_model axis: (out, 576) → (out, 768)."""
        if self.P_dmodel is not None:
            P = self.P_dmodel.to(W.device)
            return W @ P
        # Fallback: zero-pad
        out = torch.zeros(W.shape[0], self.d_model_tgt, device=W.device)
        out[:, :W.shape[1]] = W
        return out

    def _project_embedding(self, W: Tensor) -> Tensor:
        """Project embedding: (49152, 576) → (50257, 768)."""
        # Project d_model
        if self.P_dmodel is not None:
            P = self.P_dmodel.to(W.device)
            W_proj = W @ P  # (49152, 768)
        else:
            W_proj = torch.zeros(W.shape[0], self.d_model_tgt, device=W.device)
            W_proj[:, :W.shape[1]] = W

        # Pad vocab
        if W_proj.shape[0] < self.vocab_tgt:
            pad = torch.zeros(self.vocab_tgt - W_proj.shape[0], self.d_model_tgt)
            W_proj = torch.cat([W_proj, pad], dim=0)

        return W_proj

    def _convert_norm(self, W: Tensor) -> Tensor:
        """Convert RMSNorm gamma → LayerNorm gamma (same shape, just project d_model)."""
        if W.shape[0] == self.d_model_src:
            # Pad to target d_model
            out = torch.zeros(self.d_model_tgt, device=W.device)
            out[:W.shape[0]] = W
            return out
        return W

    def _extract_qkv(self, l_src: Dict) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract and project Q, K, V from source layer."""
        q_w = l_src[FunctionalRole.CONTEXT_QUERY].tensor.float()   # (n_heads*head_dim, 576)
        k_w = l_src[FunctionalRole.CONTEXT_KEY].tensor.float()     # (n_kv*head_dim, 576)
        v_w = l_src[FunctionalRole.CONTEXT_VALUE].tensor.float()   # (n_kv*head_dim, 576)

        # Project d_model: (x, 576) → (x, 768)
        q_proj = self._project_dmodel(q_w)  # (576, 768)
        k_proj = self._project_dmodel(k_w)  # (192, 768)
        v_proj = self._project_dmodel(v_w)  # (192, 768)

        # Repeat KV heads to match GPT-2's MHA (12 heads for Q, K, V each)
        # SmolLM: 9 Q heads, 3 KV heads → GPT-2: 12 heads
        # Map: head i in GPT-2 uses KV head (i % 3) from source
        q_full = self._remap_heads(q_proj, self.n_heads_src, self.n_heads_tgt)   # (768, 768)
        k_full = self._remap_heads(k_proj, 3, self.n_heads_tgt)                  # (768, 768)
        v_full = self._remap_heads(v_proj, 3, self.n_heads_tgt)                  # (768, 768)

        return q_full, k_full, v_full

    def _remap_heads(self, W: Tensor, src_heads: int, tgt_heads: int) -> Tensor:
        """
        Remap attention heads from src_heads to tgt_heads.
        W: (src_heads * head_dim, d_model)
        Returns: (tgt_heads * head_dim, d_model)
        """
        head_dim = self.head_dim
        W_3d = W.reshape(src_heads, head_dim, -1)  # (src_heads, head_dim, d_model)

        if src_heads == tgt_heads:
            return W.reshape(tgt_heads * head_dim, -1)

        # Select heads cyclically
        selected = []
        for i in range(tgt_heads):
            src_idx = i % src_heads
            selected.append(W_3d[src_idx])

        return torch.stack(selected, dim=0).reshape(tgt_heads * head_dim, -1)

    def _fuse_qkv(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Fuse separate Q, K, V into GPT-2's c_attn format.
        q: (768, 768), k: (768, 768), v: (768, 768)
        Returns: (768, 2304) — [Q | K | V] concatenated along output dim
        """
        return torch.cat([q, k, v], dim=0).T  # (2304, 768).T = (768, 2304)

    def _convert_ffn_up(self, gate: Tensor, up: Tensor) -> Tensor:
        """
        Convert SwiGLU gate+up → GELU c_fc.
        gate: (1536, 576), up: (1536, 576)
        GPT-2 c_fc: (768, 3072)
        Strategy: concatenate gate+up → project to d_inner_tgt
        """
        # Concatenate: (1536+1536, 576) = (3072, 576)
        concat = torch.cat([gate, up], dim=0)  # (3072, 576)

        # Project d_model: (3072, 576) → (3072, 768)
        proj = self._project_dmodel(concat.T).T  # (3072, 768)

        # Truncate/pad to target d_inner (3072)
        if proj.shape[0] > self.d_inner_tgt:
            proj = proj[:self.d_inner_tgt]
        elif proj.shape[0] < self.d_inner_tgt:
            pad = torch.zeros(self.d_inner_tgt - proj.shape[0], self.d_model_tgt)
            proj = torch.cat([proj, pad], dim=0)

        return proj  # (3072, 768)

    def _convert_ffn_down(self, down: Tensor) -> Tensor:
        """
        Convert SwiGLU down → GELU c_proj.
        down: (576, 1536)
        GPT-2 c_proj: (3072, 768)
        """
        # Project: (576, 1536) → (768, 3072)
        proj = self._project_dmodel(down.T).T  # (1536, 768)

        # Pad to d_inner_tgt
        if proj.shape[0] < self.d_inner_tgt:
            pad = torch.zeros(self.d_inner_tgt - proj.shape[0], self.d_model_tgt)
            proj = torch.cat([proj, pad], dim=0)
        elif proj.shape[0] > self.d_inner_tgt:
            proj = proj[:self.d_inner_tgt]

        return proj  # (3072, 768)
