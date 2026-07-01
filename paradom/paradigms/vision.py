from typing import Dict
from paradom.paradigms.base import BaseParadigmHandler
from paradom.core.taxonomy import FunctionalRole

class VisionParadigmHandler(BaseParadigmHandler):
    """
    Handler for Computer Vision models (CNN, ViT).
    """

    @property
    def paradigm_name(self) -> str:
        return "vision"

    @property
    def layer_roles(self) -> Dict[str, FunctionalRole]:
        return {
            "conv": FunctionalRole.SPATIAL_FILTER,
            "patch_embed": FunctionalRole.PATCH_EMBED,
            "attn.qkv": FunctionalRole.CONTEXT_QUERY, # Combined QKV in some ViTs
            "attn.proj": FunctionalRole.CONTEXT_OUTPUT,
            "mlp.fc1": FunctionalRole.FFN_EXPAND,
            "mlp.fc2": FunctionalRole.FFN_CONTRACT,
            "bn": FunctionalRole.NORMALIZATION,
            "head": FunctionalRole.OUTPUT_HEAD,
        }

    def validate_equivalence(self, source_role: FunctionalRole, target_role: FunctionalRole) -> bool:
        """
        Vision specific: filters and patch embeds are functionally related spatial features.
        """
        spatial_roles = {
            FunctionalRole.SPATIAL_FILTER, 
            FunctionalRole.PATCH_EMBED
        }
        
        if source_role == target_role:
            return True
            
        if source_role in spatial_roles and target_role in spatial_roles:
            return True # Can potentially swap spatial filters into patch embeds
            
        return False
