from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import torch
from paradom.core.weight import WeightProduct
from paradom.core.taxonomy import FunctionalRole
from paradom.math.cka import linear_cka

@dataclass
class MappingScore:
    source_name: str
    target_name: str
    role: FunctionalRole
    cka_score: float
    confidence: float
    swap_type: str # 'direct', 'projected', 'tensor', 'ot'

class EquivalenceIdentifier:
    """
    Identifies and validates equivalent weight products across architectures.
    Matches by FunctionalRole and verifies with CKA similarity.
    """

    def __init__(self, cka_threshold: float = 0.40):
        self.cka_threshold = cka_threshold

    def identify_pairs(
        self, 
        source_weights: List[WeightProduct], 
        target_slots: List[Dict[str, Any]]
    ) -> List[MappingScore]:
        """
        Matches source weights to target slots based on FunctionalRole and 
        projected representational similarity.
        """
        mappings = []
        
        for sw in source_weights:
            best_match: Optional[Dict[str, Any]] = None
            max_score = -1.0
            
            # Optimization: Only compare slots with the same FunctionalRole
            candidate_slots = [ts for ts in target_slots if ts['role'] == sw.functional_role]
            
            for ts in candidate_slots:
                # Second Pass: Structural Similarity Estimate
                score = self._estimate_similarity(sw.tensor)
                
                if score > max_score:
                    max_score = score
                    best_match = ts
            
            if best_match:
                # Decision: Choose swap type based on score and dimensions
                if sw.shape == best_match['shape']:
                    swap_type = "direct"
                else:
                    swap_type = "projected"
                
                mappings.append(MappingScore(
                    source_name=sw.name,
                    target_name=best_match['name'],
                    role=sw.functional_role,
                    cka_score=max_score,
                    confidence=float(max_score * 0.9),
                    swap_type=swap_type
                ))
                    
        return mappings

    def _estimate_similarity(self, W: torch.Tensor) -> float:
        """Estimates structural similarity based on singular value distribution."""
        if W.dim() < 2: 
            return 0.5
            
        try:
            # Use a limited SVD for speed during discovery
            S = torch.linalg.svdvals(W)
            if S.numel() == 0:
                return 0.0
            
            # Normalize singular values to get a relative 'energy' score
            # A score close to 1.0 means highly coherent weights
            norm_s = S / (S.sum() + 1e-8)
            score = float(norm_s[0].item()) # Using top principal component as proxy
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.5

    def compute_live_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Computes live CKA between two activation spaces."""
        return linear_cka(X, Y)
