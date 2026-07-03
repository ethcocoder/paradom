import pytest
import torch
from paradom.core.cka import linear_cka, weight_cka
from paradom.core.ssm_derivation import (
    derive_A_log_from_attention,
    derive_D_from_value_proj,
    derive_conv1d_from_attention,
)
from paradom.mappings.tiny_transformer_to_mamba import TinyTransformerToMambaMapper
from paradom.models.tiny_transformer import TinyTransformer


def test_linear_cka_identical():
    X = torch.randn(32, 16)
    assert linear_cka(X, X) > 0.99


def test_linear_cka_orthogonal():
    X = torch.eye(8)
    Y = torch.flip(torch.eye(8), dims=[1])
    assert linear_cka(X, Y) < 0.5


def test_derive_A_log_shape():
    q = torch.randn(256, 256)
    k = torch.randn(256, 256)
    A_log = derive_A_log_from_attention(q, k, d_inner=512, d_state=16)
    assert A_log.shape == (512, 16)


def test_derive_D_shape():
    v = torch.randn(256, 256)
    D = derive_D_from_value_proj(v, d_inner=512)
    assert D.shape == (512,)


def test_derive_conv1d_shape():
    v = torch.randn(256, 256)
    conv = derive_conv1d_from_attention(v, d_inner=512, d_conv=4)
    assert conv.shape == (512, 1, 4)


def test_tiny_mapper_produces_full_mamba_state():
    model = TinyTransformer()
    state = model.state_dict()
    mapper = TinyTransformerToMambaMapper()
    target, eq_map = mapper.convert(state, swap_fraction=1.0)

    expected_keys = {
        "embedding.weight",
        "lm_head.weight",
        "norm.weight",
        "layers.0.in_proj.weight",
        "layers.0.A_log",
        "layers.0.D",
        "layers.0.conv1d.weight",
        "layers.1.dt_proj.bias",
    }
    for key in expected_keys:
        assert key in target, f"Missing {key}"

    assert len(eq_map.pairs) >= 10
    assert eq_map.mean_cka > 0.0
    assert len(eq_map.unmapped_source) >= 1  # post_attention_layernorm unused


def test_mapper_swap_fraction_reduces_identity_cka():
    model = TinyTransformer()
    state = model.state_dict()
    mapper = TinyTransformerToMambaMapper()
    full, _ = mapper.convert(state, swap_fraction=1.0)
    partial, _ = mapper.convert(state, swap_fraction=0.1)
    diff = (full["embedding.weight"] - partial["embedding.weight"]).abs().mean()
    assert diff > 0.0
