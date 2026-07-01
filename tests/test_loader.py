import pytest
import torch
from paradom.core.loader import ModelLoader
from paradom.core.enums import FunctionalRole

def test_model_loader_mock_dict():
    loader = ModelLoader()
    mock_weights = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
        "model.embed_tokens.weight": torch.randn(1000, 128)
    }
    
    products = list(loader.stream_layers(mock_weights, architecture="llama"))
    
    assert len(products) == 3
    
    # Check roles
    for p in products:
        if "q_proj" in p.name:
            assert p.functional_role == FunctionalRole.CONTEXT_QUERY
            assert p.layer_index == 0
        elif "embed_tokens" in p.name:
            assert p.functional_role == FunctionalRole.EMBEDDING
