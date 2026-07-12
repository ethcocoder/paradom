import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from .enums import FunctionalRole
from .types import WeightProduct


class SubspaceProjector:
    """
    Language-aware subspace projectors for d_model and d_inner axes.

    Instead of SVD truncation (uniformly degrades all features), this keeps
    the most important subspaces entirely and drops the least important ones.

    "If an AI knows English and Amharic, when breaking down dimensions,
    remove one language entirely rather than degrading both."

    Supports two methods:
    - "svd": Original SVD-based projection from weight matrices (fast, no calibration)
    - "functional": Functional importance profiling via calibration data (better quality)
    """

    def __init__(self):
        self.P_dmodel: Optional[Tensor] = None
        self.P_dinner: Optional[Tensor] = None
        self.src_dmodel: Optional[int] = None
        self.tgt_dmodel: Optional[int] = None
        self.src_dinner: Optional[int] = None
        self.tgt_dinner: Optional[int] = None
        self.method: str = "svd"
        self.functional_importance: Optional[Tensor] = None

    def compute(
        self,
        source_products: List[WeightProduct],
        target_config: Dict[str, any],
        method: str = "svd",
        calibration_data: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Compute projectors from source weight matrices.

        Args:
            method: "svd" (default, fast) or "functional" (uses calibration data)
            calibration_data: Tensor of shape (n_tokens, d_model) from calibration runs

        P_dmodel: from embedding matrix SVD — the top-k right singular vectors
        represent the most important "languages" (subspaces) the model uses.
        """
        self.method = method

        for wp in source_products:
            if wp.functional_role == FunctionalRole.EMBEDDING:
                self.src_dmodel = wp.tensor.shape[1]
                break
        if self.src_dmodel is None:
            self.src_dmodel = source_products[0].tensor.shape[-1]

        self.tgt_dmodel = target_config['d_model']

        if self.src_dmodel > self.tgt_dmodel:
            if method == "functional" and calibration_data is not None:
                self._compute_functional_dmodel(calibration_data)
            else:
                self._compute_svd_dmodel(source_products)

        for wp in source_products:
            if wp.functional_role == FunctionalRole.FFN_GATE:
                self.src_dinner = wp.tensor.shape[0]
                break

        self.tgt_dinner = target_config.get('d_inner', self.src_dinner)

        if self.src_dinner is not None and self.src_dinner > self.tgt_dinner:
            if method == "functional" and calibration_data is not None:
                self._compute_functional_dinner(source_products, calibration_data)
            else:
                self._compute_svd_dinner(source_products)

        return self.P_dmodel, self.P_dinner

    def _compute_svd_dmodel(self, source_products):
        """Original SVD-based d_model projection."""
        for wp in source_products:
            if wp.functional_role == FunctionalRole.EMBEDDING:
                W = wp.tensor.float()
                _, _, Vh = torch.linalg.svd(W, full_matrices=False)
                self.P_dmodel = Vh[:self.tgt_dmodel, :].T
                break

    def _compute_svd_dinner(self, source_products):
        """Original SVD-based d_inner projection."""
        for wp in source_products:
            if wp.functional_role == FunctionalRole.FFN_GATE:
                W = wp.tensor.float()
                rank = min(W.shape)
                need_full = self.tgt_dinner > rank
                U, _, _ = torch.linalg.svd(W, full_matrices=need_full)
                self.P_dinner = U[:, :self.tgt_dinner]
                break

    def _compute_functional_dmodel(self, calibration_data: Tensor):
        """
        PCA on calibration data for d_model projection.

        Instead of SVD on a single weight matrix, this uses the model's
        actual hidden states to find the most important directions.
        """
        H = calibration_data.float()  # (n_tokens, d_model)
        H_centered = H - H.mean(dim=0)

        # Covariance matrix
        cov = (H_centered.T @ H_centered) / (H_centered.shape[0] - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        sorted_idx = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_idx]

        # Take top target_dim eigenvectors as projection
        self.P_dmodel = eigenvectors[:, :self.tgt_dmodel]

        # Store importance scores for diagnostics
        self.functional_importance = eigenvalues[sorted_idx]

        print(f"  [Subspace] Functional d_model projection: {self.src_dmodel} → {self.tgt_dmodel}")
        print(f"  [Subspace] Top eigenvalue ratio: {eigenvalues[sorted_idx[0]] / eigenvalues[sorted_idx[-1]]:.1f}x")

    def _compute_functional_dinner(self, source_products, calibration_data: Tensor):
        """
        PCA on gate_proj for d_inner projection.
        Uses the same calibration data but projects through the gate matrix.
        """
        # Find gate_proj matrix
        gate_W = None
        for wp in source_products:
            if wp.functional_role == FunctionalRole.FFN_GATE:
                gate_W = wp.tensor.float()
                break

        if gate_W is None:
            return

        # Project calibration data through gate matrix to get d_inner activations
        # calibration_data is (n_tokens, d_model), gate_W is (d_inner, d_model)
        # So we need to multiply: H @ gate_W.T → (n_tokens, d_inner)
        H_dinner = calibration_data.float() @ gate_W.T  # (n_tokens, d_inner)
        H_centered = H_dinner - H_dinner.mean(dim=0)

        # Covariance
        cov = (H_centered.T @ H_centered) / (H_centered.shape[0] - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        sorted_idx = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_idx]

        # Take top target_dim eigenvectors
        self.P_dinner = eigenvectors[:, :self.tgt_dinner]

        print(f"  [Subspace] Functional d_inner projection: {self.src_dinner} → {self.tgt_dinner}")


def apply_subspace_projectors(
    wp: WeightProduct,
    subspace: SubspaceProjector
) -> Tensor:
    """
    Apply d_model and d_inner projectors to a weight tensor.

    For each weight, determines which axes correspond to d_model/d_inner
    and applies the shared projector P.

    Returns the projected weight (float32).
    """
    W = wp.tensor.float()
    role = wp.functional_role

    if role == FunctionalRole.EMBEDDING:
        if subspace.P_dmodel is not None:
            W = W @ subspace.P_dmodel
        return W

    if role == FunctionalRole.OUTPUT_HEAD:
        if subspace.P_dmodel is not None:
            W = W @ subspace.P_dmodel
        return W

    if role in (FunctionalRole.NORMALIZATION, FunctionalRole.POST_NORMALIZATION,
                FunctionalRole.FINAL_NORMALIZATION):
        if subspace.P_dmodel is not None:
            W = subspace.P_dmodel.T @ W
        return W

    if role in (FunctionalRole.CONTEXT_QUERY, FunctionalRole.CONTEXT_KEY,
                FunctionalRole.CONTEXT_VALUE):
        if subspace.P_dmodel is not None:
            W = W @ subspace.P_dmodel
        return W

    if role == FunctionalRole.CONTEXT_OUTPUT:
        if subspace.P_dmodel is not None:
            W = subspace.P_dmodel.T @ W
        return W

    if role in (FunctionalRole.FFN_GATE, FunctionalRole.FFN_EXPAND):
        if subspace.P_dinner is not None:
            W = subspace.P_dinner.T @ W
        if subspace.P_dmodel is not None:
            W = W @ subspace.P_dmodel
        return W

    if role == FunctionalRole.FFN_CONTRACT:
        if subspace.P_dmodel is not None:
            W = subspace.P_dmodel.T @ W
        if subspace.P_dinner is not None:
            W = W @ subspace.P_dinner
        return W

    return W
