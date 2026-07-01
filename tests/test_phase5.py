import pytest
from paradom.core.validator import Validator
import os

def test_phase5_validator_report():
    """Verify Day 5 Quality Report generation."""
    source_path = "tests/data/mock_llama.safetensors"
    swapped_path = "tests/data/swapped_model.safetensors"
    
    # Ensure swapped model exists from previous test or create it
    if not os.path.exists(swapped_path):
        import torch
        from safetensors.torch import save_file
        save_file({"target.layers.0.q": torch.randn(32, 32)}, swapped_path)
    
    validator = Validator(source_path, swapped_path)
    report = validator.run_validation()
    
    assert "overall_quality" in report
    assert "mean_cka" in report
    assert report["mean_cka"] >= 0.0
    assert report["overall_quality"] in ["EXCELLENT", "GOOD", "ACCEPTABLE", "DEGRADED"]
