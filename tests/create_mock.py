import torch
from safetensors.torch import save_file
import os

def create_mock_model(path: str):
    """Creates a tiny 2-layer Llama-like mock model for testing."""
    tensors = {
        "model.embed_tokens.weight": torch.randn(100, 32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(32, 32),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(32, 32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 32),
        "model.layers.0.mlp.up_proj.weight": torch.randn(64, 32),
        "model.layers.0.mlp.down_proj.weight": torch.randn(32, 64),
        "model.layers.0.input_layernorm.weight": torch.randn(32),
        "model.norm.weight": torch.randn(32),
        "lm_head.weight": torch.randn(100, 32),
    }
    save_file(tensors, path)

if __name__ == "__main__":
    os.makedirs("tests/data", exist_ok=True)
    create_mock_model("tests/data/mock_llama.safetensors")
    print("Mock model created at tests/data/mock_llama.safetensors")
