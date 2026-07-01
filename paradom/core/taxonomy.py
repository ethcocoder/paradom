from enum import Enum

class FunctionalRole(Enum):
    """Universal taxonomy for neural network weight roles."""
    # Universal roles
    EMBEDDING          = "embedding"           # Input representation
    OUTPUT_HEAD        = "output_head"         # Final prediction layer
    NORMALIZATION      = "normalization"       # Scale/shift parameters
    BIAS               = "bias"               # Additive offset terms

    # Context & attention roles
    CONTEXT_QUERY      = "context_query"       # Q matrices, SSM B proj
    CONTEXT_KEY        = "context_key"         # K matrices, SSM C proj
    CONTEXT_VALUE      = "context_value"       # V matrices, value aggregation
    CONTEXT_OUTPUT     = "context_output"      # Output projection

    # Feed-forward roles
    FFN_EXPAND         = "ffn_expand"          # Up/gate projections
    FFN_CONTRACT       = "ffn_contract"        # Down projections

    # Vision-specific roles
    SPATIAL_FILTER     = "spatial_filter"      # CNN convolutional filters
    PATCH_EMBED        = "patch_embed"         # ViT patch embedding

    # RL-specific roles
    STATE_ENCODER      = "state_encoder"       # Feature extraction layers
    ACTION_HEAD        = "action_head"         # Policy output layers
    VALUE_HEAD         = "value_head"          # Value estimation layers

    # Generative-specific roles
    NOISE_PREDICTOR    = "noise_predictor"     # Diffusion UNet core
    FLOW_PREDICTOR     = "flow_predictor"      # Flow matching vector field
    TIME_EMBED         = "time_embed"          # Timestep conditioning

    # Graph-specific roles
    NODE_TRANSFORM     = "node_transform"      # Node feature transformation
    EDGE_TRANSFORM     = "edge_transform"      # Edge feature transformation
    AGGREGATION        = "aggregation"         # Message aggregation weights
