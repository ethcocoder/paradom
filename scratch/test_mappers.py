import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paradom.mappings import MAPPING_REGISTRY
from paradom.core.types import WeightProduct
from paradom.core.enums import FunctionalRole

def test_mappers():
    # 1. Test Dense to MoE
    print("\n--- Testing Dense to MoE Mapper ---")
    mapper_cls = MAPPING_REGISTRY[("llama", "mixtral")]
    mapper = mapper_cls()
    
    # Mock some dense parameters
    d_model = 1024
    d_inner = 4096
    
    w_up = WeightProduct(
        name="layers.0.mlp.up_proj.weight",
        tensor=torch.randn(d_inner, d_model),
        shape=(d_inner, d_model),
        functional_role=FunctionalRole.FFN_EXPAND,
        paradigm="transformer",
        architecture="llama",
        layer_index=0
    )
    
    target_config = {
        "d_model": d_model,
        "d_inner": d_inner,
        "num_experts": 8
    }
    
    target_dict, map_info = mapper.convert([w_up], target_config)
    print(f"Generated {len(target_dict)} target tensors.")
    assert "layers.0.block_sparse_moe.gate.weight" in target_dict
    assert target_dict["layers.0.block_sparse_moe.gate.weight"].shape == (8, 1024)
    print("MoE Router gate initialized successfully:", target_dict["layers.0.block_sparse_moe.gate.weight"].shape)
    
    # Notice 8 experts * 1 matrix since we only gave it an UP proj
    for e in range(8):
        assert f"layers.0.block_sparse_moe.experts.{e}.w1.weight" in target_dict
    print("MoE Experts duplicated successfully.")

    # 2. Test Arch Morphing Transformer -> Transformer
    print("\n--- Testing Transformer to Transformer Mapper ---")
    mapper_cls = MAPPING_REGISTRY[("transformer", "transformer")]
    mapper = mapper_cls()
    
    target_config = {
        "d_model": 2048,
        "d_inner": 8192,
        "num_hidden_layers": 1
    }
    
    w_q = WeightProduct(
        name="layers.0.self_attn.q_proj.weight",
        tensor=torch.randn(4096, 4096),
        shape=(4096, 4096),
        functional_role=FunctionalRole.CONTEXT_QUERY,
        paradigm="transformer",
        architecture="llama",
        layer_index=0
    )
    
    target_dict, map_info = mapper.convert([w_q], target_config)
    print(f"Generated {len(target_dict)} target tensors.")
    assert "layers.0.self_attn.q_proj.weight" in target_dict
    print("Arch-morphed Attention mapping successfully transformed. Target shape:", target_dict["layers.0.self_attn.q_proj.weight"].shape)

if __name__ == '__main__':
    test_mappers()
