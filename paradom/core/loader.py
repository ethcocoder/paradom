import os
from typing import Iterator, Dict, Any, List
import torch
from safetensors import safe_open
from paradom.core.weight import WeightProduct
from paradom.core.taxonomy import FunctionalRole

class ModelLoader:
    """
    Handles lazy loading and streaming of model weights from disk.
    Ensures low-RAM usage by only keeping a small fraction of weights in memory.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_safetensors = model_path.endswith(".safetensors")
        self._weights_map = {}
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

    def stream_weights(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Streams weights one by one from the model file.
        Yields: A dictionary with a single key-value pair {name: tensor}.
        """
        if self.is_safetensors:
            with safe_open(self.model_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield {name: f.get_tensor(name)}
        else:
            # Fallback for PyTorch checkpoints (memory-heavy if not mapped)
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            for name, tensor in checkpoint.items():
                yield {name: tensor}

    def get_layer_names(self) -> List[str]:
        """Returns all weight entry names in the model."""
        if self.is_safetensors:
            with safe_open(self.model_path, framework="pt", device="cpu") as f:
                return list(f.keys())
        else:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            return list(checkpoint.keys())

class LayerBuffer:
    """
    Manages the active workspace for a single layer's conversion.
    Automatically purges tensors from RAM once the layer is processed.
    """
    
    def __init__(self, max_ram_mb: int = 2048):
        self.active_weights: Dict[str, WeightProduct] = {}
        self.max_ram_mb = max_ram_mb

    def add_weight(self, weight: WeightProduct):
        """Adds a weight to the active buffer."""
        self.active_weights[weight.name] = weight

    def clear(self):
        """Purges all active weights from RAM."""
        del self.active_weights
        self.active_weights = {}
        # Suggest garbage collection for large tensors
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
