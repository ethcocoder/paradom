import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paradom.core.types import WeightProduct
from paradom.core.enums import FunctionalRole
from paradom.mappings.transformer_to_transformer import TransformerToTransformerMapper
from paradom.core.swap_engine import SwapEngine
from transformers import AutoModelForCausalLM
import re

# Hook _condense_dim to see exactly what line dies
original_condense_dim = SwapEngine._condense_dim

def hooked_condense_dim(self, tensor, target_size, dim, depth=1):
    print(f"    [condense] Depth {depth}. incoming shape: {tensor.shape}, target: {target_size}, dim: {dim}")
    current_size = tensor.shape[dim]
    if current_size <= target_size: 
        print("    [condense] done.")
        return tensor
    
    num_merges = current_size - target_size
    print("    [condense] transpose...")
    t = tensor.transpose(0, dim).clone()
    print("    [condense] reshape...")
    t_flat = t.reshape(t.shape[0], -1).float()
    print("    [condense] norms...")
    norms = torch.norm(t_flat, p=2, dim=1, keepdim=True).clamp_min(1e-8)
    t_norm = t_flat / norms
    print("    [condense] mm...")
    sim = torch.mm(t_norm, t_norm.t())
    print("    [condense] fill_diagonal...")
    sim.fill_diagonal_(-float('inf'))
    
    k = min(sim.numel(), num_merges * 10)
    print(f"    [condense] topk k={k} on sim of shape {sim.shape}...")
    try:
        _, top_k_indices = torch.topk(sim.flatten(), k)
    except Exception as e:
        print("    [condense] topk EXCEPTION:", e)
        raise
    
    print("    [condense] iter top_k...")
    merged_rows = set()
    pairs_to_merge = []
    
    for idx in top_k_indices.tolist():
        if len(pairs_to_merge) >= num_merges: break
        
        r1 = idx // sim.shape[0]
        r2 = idx % sim.shape[0]
        
        if r1 != r2 and r1 not in merged_rows and r2 not in merged_rows:
            pairs_to_merge.append((min(r1, r2), max(r1, r2)))
            merged_rows.add(r1)
            merged_rows.add(r2)
            
    print(f"    [condense] pairs to merge: {len(pairs_to_merge)}")
    rows_to_drop = []
    for r1, r2 in pairs_to_merge:
        m1 = norms[r1].item()
        m2 = norms[r2].item()
        w_sum = m1 + m2 + 1e-8
        t[r1] = (t[r1] * (m1/w_sum)) + (t[r2] * (m2/w_sum))
        rows_to_drop.append(r2)
        
    print(f"    [condense] dropping {len(rows_to_drop)} rows")
    keep_mask = torch.ones(t.shape[0], dtype=torch.bool, device=t.device)
    keep_mask[rows_to_drop] = False
    
    print(f"    [condense] subsetting t...")
    t = t[keep_mask]
    t = t.transpose(0, dim).to(tensor.dtype)
    
    if t.shape[dim] > target_size:
        print(f"    [condense] RECURSING! t.shape is {t.shape}")
        return hooked_condense_dim(self, t, target_size, dim, depth+1)
        
    return t

SwapEngine._condense_dim = hooked_condense_dim

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", torch_dtype=torch.float32)
full_sd = {k: v.clone() for k, v in model.model.state_dict().items()}
del model

from paradom.core.matcher import FunctionalRoleMatcher
matcher = FunctionalRoleMatcher()
products = []
for key, tensor in full_sd.items():
    role = matcher.assign_role(key, "llama")
    layer_idx = -1
    m = re.search(r"layers\.(\d+)\.", key)
    if m: layer_idx = int(m.group(1))
    if "embed" in key:
        products.append(WeightProduct(
            name=key, tensor=tensor, shape=tuple(tensor.shape), functional_role=role,
            paradigm="transformer", architecture="llama", layer_index=layer_idx, dtype=tensor.dtype,
        ))

target_b = {
    "d_model": 512, "d_inner": 1408, "num_heads": 8, "num_key_value_heads": 2,
    "head_dim": 64, "num_hidden_layers": 30, "vocab_size": 49152,
}
mapper = TransformerToTransformerMapper(force_projected=False)
print("Starting swap...")

class HookedMapper(TransformerToTransformerMapper):
    def _apply_swap(self, wp, target_shape, fraction, pairs, scores, target_name=None):
        target_name = target_name or wp.name
        print(f"  --> Start {target_name} {wp.tensor.shape} -> {target_shape} ...", end="", flush=True)
        is_downscale = wp.tensor.dim() >= 1 and (
            wp.tensor.shape[0] > target_shape[0] or 
            (wp.tensor.dim() > 1 and wp.tensor.shape[1] > target_shape[1])
        )
        swap_type = __import__('paradom.core.enums', fromlist=['SwapType']).SwapType.OT if is_downscale else __import__('paradom.core.enums', fromlist=['SwapType']).SwapType.PROJECTED
        
        print(f"\nTriggering swap_engine.swap...", flush=True)
        out = self.swap_engine.swap(wp.tensor, target_shape, swap_type, None)
        return out

mapper.__class__ = HookedMapper
# only process 1 product (embed_tokens) to speed up testing
swapped_sd, eq_map = mapper.convert(products, target_b, swap_fraction=1.0)
print("\nSUCCESS!")
