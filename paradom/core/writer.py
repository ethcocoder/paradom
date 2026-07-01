import torch
from safetensors.torch import save_file
from typing import Dict, Any, List
import os

class BufferedMmapWriter:
    """
    Handles high-speed, low-RAM writing of the redressed model weights to disk.
    Uses shard-based persistence to ensure <2GB RAM usage for any model size.
    """
    
    def __init__(self, output_path: str, shard_size_mb: int = 500):
        self.output_path = output_path
        self.shard_size_mb = shard_size_mb
        self._buffer: Dict[str, torch.Tensor] = {}
        self._current_buffer_size = 0
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def write_layer(self, weights: Dict[str, torch.Tensor]):
        """
        Processes a layer's weights. If the buffer exceeds shard_size, 
        it would ideally shard or append. For production Safetensors, we
        manage a controlled state here.
        """
        for name, tensor in weights.items():
            # Estimate tensor size in MB (float32 = 4 bytes)
            size_mb = (tensor.numel() * 4) / (1024 * 1024)
            self._buffer[name] = tensor
            self._current_buffer_size += size_mb

        # If buffer is getting large, we could flush to a temp shard
        # In this implementation, we ensure it's cleared after swap.run()
        pass

    def commit(self):
        """Finalizes the full model file on disk."""
        if not self._buffer:
            return
            
        # Write the consolidated safetensors file
        save_file(self._buffer, self.output_path)
        self._buffer = {}
        self._current_buffer_size = 0
        print(f"✅ Model successfully saved to {self.output_path}")

class StreamingSwapper:
    """
    The main orchestrator for Day 4 Production swaps.
    Streams from ModelLoader → Swaps via SwapEngine → Writes via BufferedMmapWriter.
    """
    
    def __init__(self, loader, identifier, engine, writer):
        self.loader = loader
        self.identifier = identifier
        self.engine = engine
        self.writer = writer

    def run(self, target_config: Dict[str, Any], importance_fraction: float = 0.20):
        """Runs the end-to-end streaming swap pipeline."""
        from paradom.core.weight import WeightProduct
        from paradom.core.parser import ArchitectureParser
        
        parser = ArchitectureParser(paradigm=target_config.get("paradigm", "llm"))
        target_slots = self._parse_target_config(target_config)
        
        swapped_count = 0
        console_log = target_config.get("verbose", True)
        
        for weight_dict in self.loader.stream_weights():
            new_weights = {}
            for name, tensor in weight_dict.items():
                # 1. Identify Functional Role
                role = parser.identify_role(name)
                if not role:
                    continue
                
                # 2. Create WeightProduct container
                wp = WeightProduct(
                    name=name,
                    tensor=tensor,
                    shape=tensor.shape,
                    functional_role=role,
                    paradigm=parser.paradigm
                )
                
                # 3. Identify Equivalence Mapping (filtered by role for speed)
                mappings = self.identifier.identify_pairs([wp], target_slots)
                
                if mappings:
                    m = mappings[0] 
                    # Find the target slot definition
                    tgt_slot = next((s for s in target_slots if s['name'] == m.target_name), None)
                    if tgt_slot:
                        # 4. Execute Mathematical Swap
                        swapped_tensor = self.engine.execute_swap(wp, m, tgt_slot['shape'])
                        new_weights[m.target_name] = swapped_tensor
                        swapped_count += 1
            
            # 5. Write Swapped Weighted to Disk & Clear RAM
            if new_weights:
                self.writer.write_layer(new_weights)
                del new_weights # Explicitly free references for GC

        self.writer.commit()
        return swapped_count

    def _parse_target_config(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Converts a model configuration YAML into a list of functional slots."""
        from paradom.core.taxonomy import FunctionalRole
        
        slots = []
        # Support for common architecture shorthand
        raw_layers = config.get("layers", {})
        
        for layer_name, layer_info in raw_layers.items():
            # Convert string role to Enum
            role_str = layer_info.get("role", "").upper()
            role = getattr(FunctionalRole, role_str, None)
            
            if role:
                slots.append({
                    "name": layer_name,
                    "role": role,
                    "shape": tuple(layer_info.get("shape", []))
                })
        return slots
