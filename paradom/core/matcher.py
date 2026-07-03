from typing import Dict
from .enums import FunctionalRole

class FunctionalRoleMatcher:
    """
    Assigns a FunctionalRole to each weight tensor based on its
    layer name and architecture.
    """

    # Mapping from (architecture, layer_name_pattern) → FunctionalRole
    ROLE_PATTERNS: Dict[str, Dict[str, FunctionalRole]] = {
        "llama": {
            "embed_tokens":     FunctionalRole.EMBEDDING,
            "q_proj":           FunctionalRole.CONTEXT_QUERY,
            "k_proj":           FunctionalRole.CONTEXT_KEY,
            "v_proj":           FunctionalRole.CONTEXT_VALUE,
            "o_proj":           FunctionalRole.CONTEXT_OUTPUT,
            "gate_proj":        FunctionalRole.FFN_EXPAND,
            "up_proj":          FunctionalRole.FFN_EXPAND,
            "down_proj":        FunctionalRole.FFN_CONTRACT,
            "input_layernorm":  FunctionalRole.NORMALIZATION,
            "post_attention_layernorm": FunctionalRole.NORMALIZATION,
            "lm_head":          FunctionalRole.OUTPUT_HEAD,
        },
        "tinytransformer": {
            "embedding":        FunctionalRole.EMBEDDING,
            "q_proj":           FunctionalRole.CONTEXT_QUERY,
            "k_proj":           FunctionalRole.CONTEXT_KEY,
            "v_proj":           FunctionalRole.CONTEXT_VALUE,
            "o_proj":           FunctionalRole.CONTEXT_OUTPUT,
            "gate_proj":        FunctionalRole.FFN_EXPAND,
            "up_proj":          FunctionalRole.FFN_EXPAND,
            "down_proj":        FunctionalRole.FFN_CONTRACT,
            "input_layernorm":  FunctionalRole.NORMALIZATION,
            "post_attention_layernorm": FunctionalRole.NORMALIZATION,
            "lm_head":          FunctionalRole.OUTPUT_HEAD,
        },
        "tinymamba": {
            "embedding":        FunctionalRole.EMBEDDING,
            "in_proj":          FunctionalRole.CONTEXT_QUERY,
            "x_proj":           FunctionalRole.CONTEXT_KEY,
            "out_proj":         FunctionalRole.CONTEXT_OUTPUT,
            "dt_proj":          FunctionalRole.FFN_EXPAND,
            "norm":             FunctionalRole.NORMALIZATION,
            "lm_head":          FunctionalRole.OUTPUT_HEAD,
        },
        "mamba": {
            "embedding":        FunctionalRole.EMBEDDING,
            "in_proj":          FunctionalRole.CONTEXT_QUERY,
            "x_proj":           FunctionalRole.CONTEXT_KEY,
            "out_proj":         FunctionalRole.CONTEXT_OUTPUT,
            "dt_proj":          FunctionalRole.FFN_EXPAND,
            "norm":             FunctionalRole.NORMALIZATION,
            "lm_head":          FunctionalRole.OUTPUT_HEAD,
        }
    }

    def assign_role(
        self,
        layer_name: str,
        architecture: str
    ) -> FunctionalRole:
        """Assign FunctionalRole based on layer name and architecture."""
        patterns = self.ROLE_PATTERNS.get(architecture.lower(), {})
        for pattern, role in patterns.items():
            if pattern in layer_name:
                return role
        return FunctionalRole.UNKNOWN
