import pytest
import os
import yaml
from paradom.core.loader import ModelLoader
from paradom.core.equivalence import EquivalenceIdentifier
from paradom.core.swap_engine import SwapEngine
from paradom.core.writer import BufferedMmapWriter, StreamingSwapper
from paradom.core.taxonomy import FunctionalRole

def test_phase4_streaming_swapper_end_to_end():
    """Verify Day 4 Production Streaming Swap logic."""
    source_path = "tests/data/mock_llama.safetensors"
    output_path = "tests/data/swapped_model.safetensors"
    
    # Mock config
    config = {
        "paradigm": "llm",
        "layers": {
            "target.layers.0.q": {"role": "CONTEXT_QUERY", "shape": [32, 32]},
            "target.layers.0.ffn": {"role": "FFN_EXPAND", "shape": [64, 32]},
        }
    }
    
    loader = ModelLoader(source_path)
    identifier = EquivalenceIdentifier()
    engine = SwapEngine()
    writer = BufferedMmapWriter(output_path)
    
    swapper = StreamingSwapper(loader, identifier, engine, writer)
    
    # Run swap
    swapped_count = swapper.run(config)
    
    assert swapped_count > 0
    assert os.path.exists(output_path)

def test_phase4_buffered_writer_commit():
    """Verify Day 4 Writer flush/commit logic."""
    out_path = "tests/data/test_write.safetensors"
    writer = BufferedMmapWriter(out_path)
    
    import torch
    dummy_weight = {"test.weight": torch.randn(10, 10)}
    writer.write_layer(dummy_weight)
    writer.commit()
    
    assert os.path.exists(out_path)
    from safetensors.torch import load_file
    loaded = load_file(out_path)
    assert "test.weight" in loaded
