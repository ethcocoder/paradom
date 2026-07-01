from typing import Dict, Type
from paradom.mappings.base import BaseMapper

class MappingRegistry:
    """Registry to manage and retrieve mapping strategies between architectures."""
    
    _registry: Dict[str, Type[BaseMapper]] = {}

    @classmethod
    def register(cls, source_paradigm: str, target_paradigm: str, mapper_class: Type[BaseMapper]):
        key = f"{source_paradigm}_to_{target_paradigm}"
        cls._registry[key] = mapper_class

    @classmethod
    def get_mapper(cls, source_paradigm: str, target_paradigm: str) -> BaseMapper:
        key = f"{source_paradigm}_to_{target_paradigm}"
        mapper_class = cls._registry.get(key)
        if not mapper_class:
            raise ValueError(f"No mapping strategy found for {key}")
        return mapper_class()

# Register default strategies
from paradom.mappings.generic import GenericMapper

# Default LLM-to-LLM / Cross-Architecture strategy
MappingRegistry.register("llm", "llm", GenericMapper)
MappingRegistry.register("llm", "mamba", GenericMapper)
MappingRegistry.register("mamba", "llm", GenericMapper)
