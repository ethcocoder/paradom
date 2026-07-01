from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from paradom.core.taxonomy import FunctionalRole

class BaseParadigmHandler(ABC):
    """
    Abstract base class for paradigm-specific logic.
    Each paradigm (LLM, Vision, RL) must define how to parse its own structures.
    """
    
    @property
    @abstractmethod
    def paradigm_name(self) -> str:
        """Friendly name of the paradigm (e.g. 'llm', 'vision')."""
        pass

    @property
    @abstractmethod
    def layer_roles(self) -> Dict[str, FunctionalRole]:
        """Map of architecture-specific layer substrings to FunctionalRole."""
        pass

    @abstractmethod
    def validate_equivalence(self, source_role: FunctionalRole, target_role: FunctionalRole) -> bool:
        """Determines if two roles are swappable within this paradigm context."""
        pass
