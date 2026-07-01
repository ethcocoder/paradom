import pytest
import torch
import os
from paradom.core.loader import ModelLoader
from paradom.core.parser import ArchitectureParser
from paradom.core.importance import ImportanceScorer
from paradom.core.taxonomy import FunctionalRole

def test_phase1_streaming_loader():
    """Verify Day 1 Streaming Engine (ModelLoader)."""
    mock_path = "tests/data/mock_llama.safetensors"
    assert os.path.exists(mock_path), "Mock model missing. Run create_mock.py first."
    
    loader = ModelLoader(mock_path)
    count = 0
    for weight_dict in loader.stream_weights():
        for name, tensor in weight_dict.items():
            assert isinstance(name, str)
            assert isinstance(tensor, torch.Tensor)
            count += 1
    
    assert count == 11 # Total layers in mock_llama

def test_phase1_architecture_discovery():
    """Verify Day 1 Architecture Discovery (Parser)."""
    parser = ArchitectureParser(paradigm="llm")
    
    role_q = parser.identify_role("model.layers.0.self_attn.q_proj.weight")
    role_up = parser.identify_role("model.layers.0.mlp.up_proj.weight")
    role_head = parser.identify_role("lm_head.weight")
    
    assert role_q == FunctionalRole.CONTEXT_QUERY
    assert role_up == FunctionalRole.FFN_EXPAND
    assert role_head == FunctionalRole.OUTPUT_HEAD

def test_phase1_winning_ticket_scoring():
    """Verify Day 1 Winning Ticket Identification (SVD Scorer)."""
    scorer = ImportanceScorer()
    W = torch.randn(32, 32)
    
    # Compute importance mask
    mask = scorer.svd_spectrum(W, top_k_fraction=0.20)
    
    assert mask.shape == W.shape
    assert mask.max() <= 1.0
    assert mask.min() >= 0.0
    # Higher variance components should result in non-zero scores
    assert mask.mean() > 0
