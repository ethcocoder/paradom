"""Derive Mamba SSM parameters from Transformer attention weights."""

import torch
from torch import Tensor


def derive_A_log_from_attention(
    W_q: Tensor,
    W_k: Tensor,
    d_inner: int,
    d_state: int,
) -> Tensor:
    """
    Derive Mamba A_log from the attention pattern eigenstructure (Q @ K^T).

    See docs/specs/SPECIFICATIONS.md §5.1 for scientific basis.
    """
    d_model = W_q.shape[0]
    scale = (d_model ** -0.5)
    W_qk = (W_q.float() @ W_k.float().T) * scale
    eigenvalues = torch.linalg.eigvalsh(W_qk)
    top_ev = eigenvalues[-d_state:].abs().clamp(min=1e-6)
    A_log_base = torch.log(top_ev)

    row_norms = W_q.float().norm(dim=1)
    if row_norms.numel() != d_inner:
        channel_scale = torch.nn.functional.interpolate(
            row_norms.unsqueeze(0).unsqueeze(0),
            size=d_inner,
            mode="linear",
            align_corners=True,
        ).squeeze()
    else:
        channel_scale = row_norms

    channel_scale = channel_scale / channel_scale.mean().clamp(min=1e-6)
    A_log = A_log_base.unsqueeze(0) * channel_scale.unsqueeze(1)
    return A_log.to(W_q.dtype)


def derive_D_from_value_proj(W_v: Tensor, d_inner: int) -> Tensor:
    """Channel skip connection strengths from value projection row energy."""
    row_energy = W_v.float().norm(dim=1)
    if row_energy.numel() != d_inner:
        D = torch.nn.functional.interpolate(
            row_energy.unsqueeze(0).unsqueeze(0),
            size=d_inner,
            mode="linear",
            align_corners=True,
        ).squeeze()
    else:
        D = row_energy
    D = D / D.mean().clamp(min=1e-6)
    return D.to(W_v.dtype)


def derive_conv1d_from_attention(
    W_v: Tensor,
    d_inner: int,
    d_conv: int,
) -> Tensor:
    """
    Depthwise conv kernel from local mixing implied by value projection structure.
    Each channel gets a normalized box filter (local context proxy).
    """
    row_energy = W_v.float().norm(dim=1)
    if row_energy.numel() != d_inner:
        channel_energy = torch.nn.functional.interpolate(
            row_energy.unsqueeze(0).unsqueeze(0),
            size=d_inner,
            mode="linear",
            align_corners=True,
        ).squeeze()
    else:
        channel_energy = row_energy

    channel_energy = channel_energy / channel_energy.sum().clamp(min=1e-6)
    kernel = torch.full((d_inner, 1, d_conv), 1.0 / d_conv)
    kernel = kernel * channel_energy.view(d_inner, 1, 1)
    return kernel.to(W_v.dtype)
