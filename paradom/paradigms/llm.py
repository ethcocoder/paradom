from typing import Dict
from paradom.paradigms.base import BaseParadigmHandler
from paradom.core.taxonomy import FunctionalRole

class LLMParadigmHandler(BaseParadigmHandler):
    """
    Handler for Large Language Models (Transformer, Mamba, MoE).
    """

    @property
    def paradigm_name(self) -> str:
        return "llm"

    @property
    def layer_roles(self) -> Dict[str, FunctionalRole]:
        return {
            "q_proj": FunctionalRole.CONTEXT_QUERY,
            "k_proj": FunctionalRole.CONTEXT_KEY,
            "v_proj": FunctionalRole.CONTEXT_VALUE,
            "o_proj": FunctionalRole.CONTEXT_OUTPUT,
            "gate_proj": FunctionalRole.FFN_EXPAND,
            "up_proj": FunctionalRole.FFN_EXPAND,
            "down_proj": FunctionalRole.FFN_CONTRACT,
            # Mamba/SSM names
            "in_proj": FunctionalRole.CONTEXT_QUERY, # Usually combined in SSM
            "x_proj": FunctionalRole.CONTEXT_QUERY,
            "dt_proj": FunctionalRole.FFN_CONTRACT,
            "out_proj": FunctionalRole.CONTEXT_OUTPUT,
            # Common
            "embed_tokens": FunctionalRole.EMBEDDING,
            "norm": FunctionalRole.NORMALIZATION,
            "lm_head": FunctionalRole.OUTPUT_HEAD,
        }

    def validate_equivalence(self, source_role: FunctionalRole, target_role: FunctionalRole) -> bool:
        """
        In LLMs, Query/Key/Value roles are interchangeable contexts,
        and Expand/Contract roles are interchangeable FFN parts.
        """
        # Groupings that are "swappable" in a cross-arch sense
        context_roles = {
            FunctionalRole.CONTEXT_QUERY, 
            FunctionalRole.CONTEXT_KEY, 
            FunctionalRole.CONTEXT_VALUE
        }
        ffn_roles = {
            FunctionalRole.FFN_EXPAND, 
            FunctionalRole.FFN_CONTRACT
        }
        
        if source_role == target_role:
            return True
        
        if source_role in context_roles and target_role in context_roles:
            return True
            
        if source_role in ffn_roles and target_role in ffn_roles:
            return True
            
        return False
