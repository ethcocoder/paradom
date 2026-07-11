from .tiny_transformer_to_mamba import TinyTransformerToMambaMapper
from .transformer_to_mamba import TransformerToMambaMapper
from .dense_to_moe import DenseToMoEMapper
from .transformer_to_transformer import TransformerToTransformerMapper

MAPPING_REGISTRY = {
    ("tinytransformer", "tinymamba"): TinyTransformerToMambaMapper,
    ("llama", "mamba"): TransformerToMambaMapper,
    ("mistral", "mamba"): TransformerToMambaMapper,
    ("transformer", "mamba"): TransformerToMambaMapper,
    ("llama", "mixtral"): DenseToMoEMapper,
    ("transformer", "moe"): DenseToMoEMapper,
    ("llama", "llama"): TransformerToTransformerMapper,
    ("transformer", "transformer"): TransformerToTransformerMapper,
}
