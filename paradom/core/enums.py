from enum import Enum

class FunctionalRole(Enum):
    """Universal taxonomy of weight functional roles across all paradigms."""
    # Universal
    EMBEDDING         = "embedding"
    OUTPUT_HEAD       = "output_head"
    NORMALIZATION     = "normalization"
    POST_NORMALIZATION = "post_normalization"
    FINAL_NORMALIZATION = "final_normalization"
    BIAS              = "bias"
    # Context / Attention
    CONTEXT_QUERY     = "context_query"
    CONTEXT_KEY       = "context_key"
    CONTEXT_VALUE     = "context_value"
    CONTEXT_OUTPUT    = "context_output"
    # Feed-forward
    FFN_EXPAND        = "ffn_expand"
    FFN_GATE          = "ffn_gate"
    FFN_CONTRACT      = "ffn_contract"
    # Vision
    SPATIAL_FILTER    = "spatial_filter"
    PATCH_EMBED       = "patch_embed"
    # RL
    STATE_ENCODER     = "state_encoder"
    ACTION_HEAD       = "action_head"
    VALUE_HEAD        = "value_head"
    # Generative
    NOISE_PREDICTOR   = "noise_predictor"
    FLOW_PREDICTOR    = "flow_predictor"
    TIME_EMBED        = "time_embed"
    # Graph
    NODE_TRANSFORM    = "node_transform"
    EDGE_TRANSFORM    = "edge_transform"
    AGGREGATION       = "aggregation"
    # Unknown (must be resolved before swapping)
    UNKNOWN           = "unknown"

class SwapType(Enum):
    DIRECT      = "direct"       # Same shape, high CKA — direct copy
    PROJECTED   = "projected"    # Different shape — SVD projection
    TENSOR      = "tensor"       # Multi-dim (CNN) — Tucker decomp
    OT          = "ot"           # Weak equivalence — optimal transport
    DERIVED     = "derived"      # No source equivalent — mathematically derived
    SKIP        = "skip"         # No equivalent — Xavier init

class QualityTier(Enum):
    EXCELLENT  = "excellent"     # ≥85% retention
    GOOD       = "good"          # 70–85% retention
    ACCEPTABLE = "acceptable"    # 55–70% retention
    DEGRADED   = "degraded"      # <55% retention
