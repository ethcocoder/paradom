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

    P_dmodel is computed from the embedding matrix SVD — the embedding
    captures the true semantic structure of the model's hidden space.
    P_dinner is computed from the first gate_proj SVD.
    """

    def __init__(self):
        self.P_dmodel: Optional[Tensor] = None
        self.P_dinner: Optional[Tensor] = None
        self.src_dmodel: Optional[int] = None
        self.tgt_dmodel: Optional[int] = None
        self.src_dinner: Optional[int] = None
        self.tgt_dinner: Optional[int] = None

    def compute(
        self,
        source_products: List[WeightProduct],
        target_config: Dict[str, any]
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Compute projectors from source weight matrices.

        P_dmodel: from embedding matrix SVD — the top-k right singular vectors
        represent the most important "languages" (subspaces) the model uses.
        """
        for wp in source_products:
            if wp.functional_role == FunctionalRole.EMBEDDING:
                self.src_dmodel = wp.tensor.shape[1]
                break
        if self.src_dmodel is None:
            self.src_dmodel = source_products[0].tensor.shape[-1]

        self.tgt_dmodel = target_config['d_model']

        if self.src_dmodel > self.tgt_dmodel:
            for wp in source_products:
                if wp.functional_role == FunctionalRole.EMBEDDING:
                    W = wp.tensor.float()
                    _, _, Vh = torch.linalg.svd(W, full_matrices=False)
                    self.P_dmodel = Vh[:self.tgt_dmodel, :].T
                    break

        for wp in source_products:
            if wp.functional_role == FunctionalRole.FFN_GATE:
                self.src_dinner = wp.tensor.shape[0]
                break

        self.tgt_dinner = target_config.get('d_inner', self.src_dinner)

        if self.src_dinner is not None and self.src_dinner > self.tgt_dinner:
            for wp in source_products:
                if wp.functional_role == FunctionalRole.FFN_GATE:
                    W = wp.tensor.float()
                    rank = min(W.shape)
                    need_full = self.tgt_dinner > rank
                    U, _, _ = torch.linalg.svd(W, full_matrices=need_full)
                    self.P_dinner = U[:, :self.tgt_dinner]
                    break

        return self.P_dmodel, self.P_dinner


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
