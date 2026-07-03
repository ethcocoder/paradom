from .tiny_transformer_to_mamba import TinyTransformerToMambaMapper
from .transformer_to_mamba import TransformerToMambaMapper

MAPPING_REGISTRY = {
    ("tinytransformer", "tinymamba"): TinyTransformerToMambaMapper,
    ("llama", "mamba"): TransformerToMambaMapper,
    ("mistral", "mamba"): TransformerToMambaMapper,
    ("transformer", "mamba"): TransformerToMambaMapper,
}
