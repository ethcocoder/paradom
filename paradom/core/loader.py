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

    def stream_metadata(
        self,
        source: Any,
        architecture: str,
        paradigm: str = "llm"
    ) -> Generator[WeightProduct, None, None]:
        """
        Yields WeightProducts with empty (zero) tensors for structure identification.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix == ".safetensors":
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        # We use metadata only
                        slice = f.get_slice(key)
                        # Create a zero tensor of the correct shape/dtype for metadata
                        metadata_tensor = torch.empty(0) 
                        yield self._build_product(key, metadata_tensor, architecture, paradigm, shape=slice.get_shape())
                return
            else:
                # For PyTorch files, we unfortunately usually have to load the dict keys
                # but we can try to avoid loading the actual storage if it's a mmap
                weights = torch.load(source, map_location="cpu")
                if "model" in weights: weights = weights["model"]
                for key, tensor in weights.items():
                    yield self._build_product(key, torch.empty(0), architecture, paradigm, shape=tuple(tensor.shape))
        else:
            # Fallback to full load if it's already a dict
            for wp in self.stream_layers(source, architecture, paradigm):
                yield wp

    def stream_layers(
        self,
        source: Any,
        architecture: str,
        paradigm: str = "llm"
    ) -> Generator[WeightProduct, None, None]:
        """
        Generator that yields WeightProduct objects with full tensors.
        """
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

    def _build_product(self, name: str, tensor: torch.Tensor, architecture: str, paradigm: str, shape: Optional[tuple] = None) -> WeightProduct:
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
            shape=shape or tuple(tensor.shape),
            functional_role=role,
            paradigm=paradigm,
            architecture=architecture,
            layer_index=layer_index,
            dtype=tensor.dtype
        )
