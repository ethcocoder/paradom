from dataclasses import dataclass
from typing import Tuple, Optional
from torch import Tensor
from paradom.core.taxonomy import FunctionalRole

@dataclass
class WeightProduct:
    """A single weight tensor with its functional metadata."""
    name: str                        # Original layer name (e.g. "layers.0.self_attn.q_proj.weight")
    tensor: Tensor                   # The raw numerical data
    shape: Tuple[int, ...]           # Tensor dimensions
    functional_role: FunctionalRole  # The universal role this weight plays
    paradigm: str                    # The paradigm it belongs to (e.g. "llm", "vision")
    importance_score: float = 0.0    # Set after ImportanceScorer analysis
    
    def __repr__(self) -> str:
        return f"WeightProduct(name='{self.name}', role={self.functional_role.value}, shape={self.shape})"
