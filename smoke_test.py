import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from paradom.core.enums import FunctionalRole, SwapType
    from paradom.core.types import WeightProduct
    from paradom.core.importance import ImportanceScorer
    from paradom.core.swap_engine import SwapEngine
    from paradom.core.loader import ModelLoader
    from paradom.core.engine import Paradom
    
    print("Successfully imported all Paradom core modules.")
    
    scorer = ImportanceScorer()
    print("ImportanceScorer initialized.")
    
    engine = SwapEngine()
    print("SwapEngine initialized.")
    
    loader = ModelLoader()
    print("ModelLoader initialized.")
    
    paradom = Paradom()
    print("Paradom orchestrator initialized.")

except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
