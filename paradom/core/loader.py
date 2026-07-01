import torch
from typing import Dict, Any, Optional, Generator
from pathlib import Path
from .types import WeightProduct
from .enums import FunctionalRole
from .matcher import FunctionalRoleMatcher

class ModelLoader:
    """
    Loads model weights from local checkpoints.
    Phase 1 focused on local PyTorch and SafeTensors.
    """

    def __init__(self, role_matcher: Optional[FunctionalRoleMatcher] = None):
        self.role_matcher = role_matcher or FunctionalRoleMatcher()

    def stream_layers(
        self,
        source: Any,  # Path string or dict (for in-memory objects in Phase 1)
        architecture: str,
        paradigm: str = "llm"
    ) -> Generator[WeightProduct, None, None]:
        """
        Generator that yields WeightProduct objects.
        """
        weights = {}
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix == ".safetensors":
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        yield self._build_product(key, tensor, architecture, paradigm)
                return
            else:
                # Assume PyTorch
                weights = torch.load(source, map_location="cpu")
                if "model" in weights: weights = weights["model"]
        elif isinstance(source, dict):
            weights = source
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        for key, tensor in weights.items():
            yield self._build_product(key, tensor, architecture, paradigm)

    def _build_product(self, name: str, tensor: torch.Tensor, architecture: str, paradigm: str) -> WeightProduct:
        role = self.role_matcher.assign_role(name, architecture)
        
        # Extract layer index if possible (e.g., layers.0. -> 0)
        layer_index = -1
        import re
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layer_index = int(match.group(1))

        return WeightProduct(
            name=name,
            tensor=tensor,
            shape=tuple(tensor.shape),
            functional_role=role,
            paradigm=paradigm,
            architecture=architecture,
            layer_index=layer_index,
            dtype=tensor.dtype
        )
