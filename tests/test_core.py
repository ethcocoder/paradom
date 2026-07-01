import pytest
import torch
from paradom.core.importance import ImportanceScorer
from paradom.core.swap_engine import SwapEngine
from paradom.core.enums import SwapType

def test_importance_scorer_svd():
    scorer = ImportanceScorer()
    # Create a simple low-rank matrix
    # W = u @ v.T (Rank 1)
    u = torch.randn(10, 1)
    v = torch.randn(10, 1)
    W = u @ v.T
    
    # 20% importance fraction
    mask = scorer.score_svd_spectrum(W, top_k_fraction=0.2)
    
    assert mask.shape == W.shape
    assert mask.dtype == torch.bool
    # Check that at least some elements are selected
    assert mask.sum() > 0
    # Check fraction (approximately due to quantile)
    fraction = mask.sum().item() / W.numel()
    assert 0.15 <= fraction <= 0.25

def test_direct_swap_success():
    engine = SwapEngine()
    W_src = torch.randn(64, 64)
    target_shape = (64, 64)
    
    W_tgt = engine.swap(W_src, target_shape, SwapType.DIRECT)
    
    assert torch.equal(W_src, W_tgt)

def test_direct_swap_with_mask():
    engine = SwapEngine()
    W_src = torch.ones(10, 10)
    target_shape = (10, 10)
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[0, 0] = True
    
    W_tgt = engine.swap(W_src, target_shape, SwapType.DIRECT, importance_mask=mask)
    
    assert W_tgt[0, 0] == 1.0
    # Other elements should be Xavier-initialized (not 1.0 with high probability)
    assert not torch.all(W_tgt == 1.0)

def test_projected_swap():
    engine = SwapEngine()
    # 1024 -> 512 projection
    W_src = torch.randn(1024, 1024)
    target_shape = (512, 512)
    
    W_tgt = engine.swap(W_src, target_shape, SwapType.PROJECTED)
    
    assert W_tgt.shape == target_shape
    # Check that energy is somewhat preserved (not all zeros)
    assert W_tgt.std() > 0
