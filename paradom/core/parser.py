import re
from typing import Dict, List, Optional
from paradom.core.taxonomy import FunctionalRole

class ArchitectureParser:
    """
    Identifies the functional role of weights based on architectural naming conventions.
    Initial focus: LLaMA (Transformer) and Mamba patterns.
    """
    
    # Common regex patterns for LLM weights
    PATTERNS = {
        FunctionalRole.CONTEXT_QUERY:  [r"q_proj", r"w_query", r"in_proj.*part_1"],
        FunctionalRole.CONTEXT_KEY:    [r"k_proj", r"w_key", r"in_proj.*part_2"],
        FunctionalRole.CONTEXT_VALUE:  [r"v_proj", r"w_value", r"in_proj.*part_3"],
        FunctionalRole.CONTEXT_OUTPUT: [r"o_proj", r"out_proj", r"mixer\.out_proj"],
        FunctionalRole.FFN_EXPAND:     [r"gate_proj", r"up_proj", r"w_gate", r"w_up", r"x_proj"],
        FunctionalRole.FFN_CONTRACT:   [r"down_proj", r"w_down", r"dt_proj"],
        FunctionalRole.EMBEDDING:      [r"embed_tokens", r"embedding", r"wte"],
        FunctionalRole.OUTPUT_HEAD:    [r"lm_head", r"classifier", r"fc_out"],
        FunctionalRole.NORMALIZATION:  [r"layernorm", r"norm", r"ln_f", r"ln_1", r"ln_2"],
    }

    def __init__(self, paradigm: str = "llm"):
        self.paradigm = paradigm

    def identify_role(self, weight_name: str) -> Optional[FunctionalRole]:
        """Guesses the functional role of a weight from its name."""
        name_lower = weight_name.lower()
        
        for role, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    return role
        
        return None

    def group_layers(self, weight_names: List[str]) -> Dict[int, List[str]]:
        """Groups weights by layer index found in the name."""
        layers = {}
        for name in weight_names:
            # Look for numbers like '.0.', '.1.', 'layer_0', etc.
            match = re.search(r"\b(\d+)\b", name)
            if match:
                idx = int(match.group(1))
                if idx not in layers:
                    layers[idx] = []
                layers[idx].append(name)
        return layers
