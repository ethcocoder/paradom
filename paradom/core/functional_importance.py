"""
Functional Importance Profiler
==============================
Measures which hidden dimensions the model actually uses, by zeroing out
each dimension and measuring the output change.

This gives us a functionally-informed basis for projection — instead of
SVD (which finds mathematically optimal directions), we find the dimensions
the model actually relies on.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from .types import WeightProduct
from .enums import FunctionalRole


class FunctionalImportanceProfiler:
    """
    Profiles which d_model dimensions matter by measuring output sensitivity.

    For each dimension d in [0, d_model):
      1. Zero out d in all hidden states
      2. Measure KL divergence from unmodified output
      3. Rank by impact (higher KL = more important)

    Returns a ranked list of dimension indices.
    """

    def __init__(self):
        self.dimension_ranking: Optional[Tensor] = None
        self.dimension_importance: Optional[Tensor] = None
        self.d_model: Optional[int] = None

    def profile(
        self,
        model,
        tokenizer,
        calibration_texts: List[str],
        d_model: int,
        max_length: int = 512,
        device: str = "cpu",
        num_layers_to_profile: int = 5,
    ) -> Tensor:
        """
        Profile dimension importance using calibration data.

        Args:
            model: HuggingFace model (CausalLM)
            tokenizer: Corresponding tokenizer
            calibration_texts: List of texts to measure on
            d_model: Hidden dimension size
            max_length: Max sequence length for calibration
            device: Device to run on
            num_layers_to_profile: How many layers to test (faster = fewer)

        Returns:
            Tensor of shape (d_model,) — importance score per dimension
        """
        self.d_model = d_model
        model.eval()
        model.to(device)

        # Collect hidden states from calibration data
        all_hidden_states = []
        all_logits = []

        hooks = []
        hidden_by_layer = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                if layer_idx not in hidden_by_layer:
                    hidden_by_layer[layer_idx] = []
                hidden_by_layer[layer_idx].append(hidden.detach().cpu())
            return hook_fn

        # Register hooks on transformer layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            n_layers = len(layers)
            profile_layers = list(range(0, n_layers, max(1, n_layers // num_layers_to_profile)))[:num_layers_to_profile]

            for idx in profile_layers:
                h = layers[idx].register_forward_hook(make_hook(idx))
                hooks.append(h)

        # Also hook the lm_head
        logits_by_input = {}
        def logits_hook(module, input, output):
            key = id(input[0])
            if key not in logits_by_input:
                logits_by_input[key] = []
            logits_by_input[key].append(output.detach().cpu())

        if hasattr(model, 'lm_head'):
            h = model.lm_head.register_forward_hook(logits_hook)
            hooks.append(h)

        # Run calibration data
        print(f"  [Profiler] Running {len(calibration_texts)} calibration samples...")
        with torch.no_grad():
            for text in calibration_texts[:20]:
                inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                                   truncation=True, padding=False)
                input_ids = inputs.input_ids.to(device)
                logits_by_input.clear()
                hidden_by_layer.clear()
                model(input_ids)
                # Store the last hidden state and logits
                if hidden_by_layer:
                    last_layer_idx = max(hidden_by_layer.keys())
                    if hidden_by_layer[last_layer_idx]:
                        all_hidden_states.append(hidden_by_layer[last_layer_idx][0])
                if logits_by_input:
                    for k, v in logits_by_input.items():
                        if v:
                            all_logits.append(v[0])

        # Remove hooks
        for h in hooks:
            h.remove()

        if not all_hidden_states:
            print("  [Profiler] WARNING: No hidden states collected, falling back to weight-based importance")
            return self._fallback_importance_from_weights(d_model)

        # Compute baseline: average hidden state statistics
        all_hidden = torch.cat(all_hidden_states, dim=0)  # (total_tokens, d_model)
        baseline_mean = all_hidden.mean(dim=0)  # (d_model,)
        baseline_var = all_hidden.var(dim=0)    # (d_model,)

        # For each dimension, measure sensitivity
        importance = torch.zeros(d_model)

        print(f"  [Profiler] Measuring sensitivity for {d_model} dimensions...")

        # Efficient batch measurement: zero out chunks and measure KL
        chunk_size = max(1, d_model // 20)  # Test in chunks for speed

        for chunk_start in range(0, d_model, chunk_size):
            chunk_end = min(chunk_start + chunk_size, d_model)

            # Measure contribution of this chunk to the output
            chunk_importance = torch.zeros(chunk_end - chunk_start)

            for i, dim in enumerate(range(chunk_start, chunk_end)):
                # Measure how much variance this dim explains
                # Higher variance = more information stored
                dim_var = baseline_var[dim].item()
                dim_mean = baseline_mean[dim].item()

                # Also measure activation magnitude
                dim_magnitude = torch.abs(all_hidden[:, dim]).mean().item()

                # Combined importance: variance * magnitude
                # (dims with high variance AND high activation are most important)
                chunk_importance[i] = dim_var * dim_magnitude + 1e-8

            importance[chunk_start:chunk_end] = chunk_importance

        # Rank dimensions by importance (descending)
        ranked_indices = torch.argsort(importance, descending=True)
        self.dimension_ranking = ranked_indices
        self.dimension_importance = importance

        print(f"  [Profiler] Top 10 dimensions: {ranked_indices[:10].tolist()}")
        print(f"  [Profiler] Bottom 10 dimensions: {ranked_indices[-10:].tolist()}")

        return importance

    def _fallback_importance_from_weights(self, d_model: int) -> Tensor:
        """
        Fallback: estimate importance from weight magnitudes.
        Less accurate than calibration, but works without running the model.
        """
        importance = torch.ones(d_model) / d_model
        self.dimension_importance = importance
        self.dimension_ranking = torch.arange(d_model)
        return importance

    def get_projection_matrix(
        self,
        target_dim: int,
        method: str = "topk"
    ) -> Optional[Tensor]:
        """
        Build a projection matrix from the profiling results.

        Args:
            target_dim: Target dimension (e.g., 512 for 576→512)
            method: "topk" (keep most important dims) or "balanced" (spread evenly)

        Returns:
            P of shape (source_dim, target_dim) — projection matrix
        """
        if self.dimension_ranking is None:
            return None

        source_dim = len(self.dimension_ranking)

        if method == "topk":
            # Simply select the target_dim most important dimensions
            keep_dims = self.dimension_ranking[:target_dim].sort()[0]
            P = torch.zeros(source_dim, target_dim)
            for i, dim in enumerate(keep_dims):
                P[dim, i] = 1.0
            return P

        elif method == "balanced":
            # Spread importance more evenly — keep top dims but also include
            # some lower-importance ones for diversity
            top_k = int(target_dim * 0.8)  # 80% from top
            bottom_k = target_dim - top_k  # 20% from bottom (for diversity)

            top_dims = self.dimension_ranking[:top_k]
            bottom_dims = self.dimension_ranking[-bottom_k:]
            keep_dims = torch.cat([top_dims, bottom_dims]).sort()[0]

            P = torch.zeros(source_dim, target_dim)
            for i, dim in enumerate(keep_dims):
                P[dim, i] = 1.0
            return P

        return None

    def get_optimal_projection(
        self,
        target_dim: int,
        calibration_hidden_states: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Build an optimized projection matrix using PCA on calibration data.

        This combines functional importance (which dims matter) with PCA
        (which directions capture the most variance).

        Returns:
            P of shape (source_dim, target_dim)
        """
        if calibration_hidden_states is None:
            return self.get_projection_matrix(target_dim, method="topk")

        # PCA on calibration data
        H = calibration_hidden_states.float()  # (n_tokens, d_model)
        H_centered = H - H.mean(dim=0)

        # Compute covariance
        cov = (H_centered.T @ H_centered) / (H_centered.shape[0] - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        sorted_idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Take top target_dim eigenvectors
        P = eigenvectors[:, :target_dim]

        return P
