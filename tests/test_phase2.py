import pytest
from paradom.paradigms.llm import LLMParadigmHandler
from paradom.paradigms.vision import VisionParadigmHandler
from paradom.core.taxonomy import FunctionalRole

def test_phase2_llm_handler():
    """Verify Day 2 LLM Paradigm Handler."""
    handler = LLMParadigmHandler()
    assert handler.paradigm_name == "llm"
    
    # Test role validation (Equivalence logic)
    # Q and K should be swappable in LLM context
    assert handler.validate_equivalence(FunctionalRole.CONTEXT_QUERY, FunctionalRole.CONTEXT_KEY) is True
    # Q and FFN_EXPAND should NOT be swappable
    assert handler.validate_equivalence(FunctionalRole.CONTEXT_QUERY, FunctionalRole.FFN_EXPAND) is False

def test_phase2_vision_handler():
    """Verify Day 2 Vision Paradigm Handler."""
    handler = VisionParadigmHandler()
    assert handler.paradigm_name == "vision"
    
    # Test role validation (Equivalence logic)
    # Conv filter and Patch embed should be swappable spatial features
    assert handler.validate_equivalence(FunctionalRole.SPATIAL_FILTER, FunctionalRole.PATCH_EMBED) is True
    # Conv filter and Normalization should NOT be swappable
    assert handler.validate_equivalence(FunctionalRole.SPATIAL_FILTER, FunctionalRole.NORMALIZATION) is False
