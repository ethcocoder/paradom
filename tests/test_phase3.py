import pytest
import torch
from paradom.math.cka import linear_cka
from paradom.math.procrustes import procrustes_projected_swap
from paradom.core.equivalence import EquivalenceIdentifier, MappingScore
from paradom.core.taxonomy import FunctionalRole
from paradom.core.weight import WeightProduct

def test_phase3_cka_similarity():
    """Verify Day 3 CKA Representational Similarity."""
    # Identical representations should have CKA near 1.0
    X = torch.randn(100, 32)
    Y = X.clone()
    score = linear_cka(X, Y)
    assert score > 0.99

    # Unrelated representations should have lower CKA
    Z = torch.randn(100, 32)
    score_unrelated = linear_cka(X, Z)
    assert score_unrelated < score

def test_phase3_procrustes_projection():
    """Verify Day 3 Procrustes Alignment (Different dimensions)."""
    W_src = torch.randn(40, 32) # (out, in)
    target_shape = (64, 48)     # Larger target
    
    W_tgt = procrustes_projected_swap(W_src, target_shape)
    
    assert W_tgt.shape == target_shape
    assert not torch.isnan(W_tgt).any()
    # Check that we preserved original variance (not all zeros)
    assert W_tgt.abs().sum() > 0

def test_phase3_equivalence_mapping():
    """Verify Day 3 Equivalence Identifier mapping."""
    identifier = EquivalenceIdentifier()
    
    sw = WeightProduct(
        name="src_q", 
        tensor=torch.randn(32, 32), 
        shape=(32, 32), 
        functional_role=FunctionalRole.CONTEXT_QUERY,
        paradigm="llm"
    )
    
    target_slots = [
        {"name": "tgt_q", "role": FunctionalRole.CONTEXT_QUERY, "shape": (32, 32)},
        {"name": "tgt_k", "role": FunctionalRole.CONTEXT_KEY, "shape": (32, 32)}
    ]
    
    mappings = identifier.identify_pairs([sw], target_slots)
    
    assert len(mappings) == 1
    assert mappings[0].target_name == "tgt_q"
    assert mappings[0].swap_type == "direct"
