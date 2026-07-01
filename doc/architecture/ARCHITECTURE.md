# System Architecture: Paradom Framework

**Document:** PARADOM-ARCH-001  
**Version:** 1.0.0  
**Date:** 2026-06-30

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Core Components](#2-core-components)
3. [Data Flow](#3-data-flow)
4. [Module Specifications](#4-module-specifications)
5. [Interface Design](#5-interface-design)
6. [Storage & State Management](#6-storage--state-management)
7. [Deployment Architecture](#7-deployment-architecture)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PARADOM FRAMEWORK                            │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │  SOURCE      │    │    PARADOM   │    │  TARGET                │ │
│  │  MODEL       │───▶│    ENGINE    │───▶│  ARCHITECTURE          │ │
│  │             │    │              │    │                        │ │
│  │ HuggingFace │    │ 1. Parse     │    │  Custom Arch           │ │
│  │ Kaggle      │    │ 2. Decompose │    │  Mamba SSM             │ │
│  │ LocalWeights│    │ 3. Map       │    │  Compressed Transformer│ │
│  │             │    │ 4. Construct │    │  MoE Architecture      │ │
│  │ LLaMA       │    │ 5. Validate  │    │  Sovereign Model       │ │
│  │ Mistral     │    │              │    │                        │ │
│  │ Falcon      │    └──────────────┘    └────────────────────────┘ │
│  │ Gemma       │                                                    │
│  └─────────────┘                                                    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     SUPPORT LAYER                           │   │
│  │   Validation Suite │ CLI │ Python API │ Benchmark Engine   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 Component Map

```
paradom/
├── core/
│   ├── loader.py           # Model weight loader (HF, GGUF, SafeTensors)
│   ├── parser.py           # Architecture parser & config extractor
│   ├── decomposer.py       # SVD, PCA, eigendecomposition engine
│   ├── mapper.py           # Cross-architecture mapping logic
│   ├── constructor.py      # Target weight construction
│   └── validator.py        # Conversion quality validator
│
├── architectures/
│   ├── base.py             # Abstract architecture definition
│   ├── transformer.py      # GPT/LLaMA/Mistral family
│   ├── mamba.py            # State Space Model (Mamba/S4)
│   ├── moe.py              # Mixture of Experts
│   └── custom.py           # User-defined architecture template
│
├── mappings/
│   ├── registry.py         # Mapping strategy registry
│   ├── attention_to_ssm.py # Transformer attn → SSM mapping
│   ├── dense_to_moe.py     # Dense FFN → MoE routing
│   ├── dim_projection.py   # Embedding dimension projection
│   └── norm_transfer.py    # LayerNorm → RMSNorm transfer
│
├── calibration/
│   ├── zero_shot.py        # No-data calibration (gradient-free)
│   ├── few_shot.py         # Calibration with small dataset
│   └── metrics.py          # Calibration quality metrics
│
├── cli/
│   └── paradom_cli.py      # Command-line interface
│
├── api/
│   └── paradom_api.py      # Python API
│
└── benchmarks/
    ├── perplexity.py        # Language modeling quality
    ├── task_bench.py        # Task-specific benchmarks
    └── similarity.py        # Representational similarity
```

---

## 3. Data Flow

### 3.1 Full Conversion Pipeline

```
INPUT: Source model path + Target architecture spec
         │
         ▼
┌─────────────────────┐
│   1. LOAD           │
│   ─────────────     │
│   • Load checkpoint │
│   • Parse config    │
│   • Detect arch     │
│   • Validate format │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   2. PARSE          │
│   ─────────────     │
│   • Extract layers  │
│   • Map layer types │
│   • Get dimensions  │
│   • Build layer DAG │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   3. DECOMPOSE      │
│   ─────────────     │
│   • SVD on attn Q/K │
│   • PCA on embeddings│
│   • Extract eigen-  │
│     structure        │
│   • Build factor map│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   4. MAP            │
│   ─────────────     │
│   • Apply mapping   │
│     strategy        │
│   • Handle dim      │
│     mismatches      │
│   • Translate each  │
│     layer group     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   5. CONSTRUCT      │
│   ─────────────     │
│   • Build target    │
│     weight tensors  │
│   • Apply bias      │
│     corrections     │
│   • Initialize      │
│     missing layers  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   6. CALIBRATE      │
│   ─────────────     │
│   (Optional)        │
│   • Zero-shot       │
│     correction      │
│   • Few-shot        │
│     alignment       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   7. VALIDATE       │
│   ─────────────     │
│   • Perplexity test │
│   • Output similarity│
│   • Task benchmarks │
│   • Quality report  │
└──────────┬──────────┘
           │
           ▼
OUTPUT: Converted weights + Quality report
```

---

## 4. Module Specifications

### 4.1 Loader Module

**Responsibility:** Load weights from any common format into a unified internal representation.

**Supported Input Formats:**
- SafeTensors (`.safetensors`) — preferred for large models
- PyTorch checkpoint (`.pt`, `.bin`)
- GGUF (Llama.cpp format)
- HuggingFace model hub (auto-download)
- Sharded checkpoints (model.safetensors.index.json)

**Output:** `ModelSnapshot` object:
```python
@dataclass
class ModelSnapshot:
    architecture: str           # "llama", "mistral", "falcon", etc.
    config: ModelConfig         # Architecture hyperparameters
    layers: Dict[str, Tensor]   # Named weight tensors
    metadata: Dict              # Training info, dtype, etc.
    total_params: int
```

---

### 4.2 Parser Module

**Responsibility:** Understand the internal structure of any loaded model.

**Key Functions:**
```python
class ArchitectureParser:
    def detect_architecture(snapshot: ModelSnapshot) -> ArchType
    def extract_layer_groups(snapshot: ModelSnapshot) -> List[LayerGroup]
    def get_attention_config(config: ModelConfig) -> AttentionConfig
    def get_ffn_config(config: ModelConfig) -> FFNConfig
    def build_layer_graph(layers: Dict) -> DAG
```

**Layer Group Types:**
- `EmbeddingGroup` — token + position embeddings
- `AttentionGroup` — Q, K, V, O projection matrices + optional bias
- `FFNGroup` — gate, up, down projections
- `NormGroup` — LayerNorm, RMSNorm parameters
- `HeadGroup` — output/unembedding layer

---

### 4.3 Decomposer Module

**Responsibility:** Apply mathematical decompositions to extract transferable structure from weights.

```python
class WeightDecomposer:
    def svd_decompose(
        W: Tensor, 
        rank: int = None,           # None = full rank
        energy_threshold: float = 0.99  # Keep 99% of signal energy
    ) -> SVDFactors

    def pca_project(
        E: Tensor,                  # Embedding matrix
        target_dim: int
    ) -> Tensor

    def eigendecompose_attention(
        W_Q: Tensor, 
        W_K: Tensor
    ) -> EigenMap

    def extract_null_space(W: Tensor) -> Tensor
```

**SVDFactors:**
```python
@dataclass
class SVDFactors:
    U: Tensor      # Left singular vectors
    S: Tensor      # Singular values (energy spectrum)
    Vh: Tensor     # Right singular vectors (transposed)
    rank: int      # Effective rank used
    energy: float  # Fraction of variance preserved
```

---

### 4.4 Mapper Module

**Responsibility:** Define and apply conversion strategies between architecture pairs.

**Mapping Registry:**
```python
MAPPING_REGISTRY = {
    ("transformer", "transformer"): SameArchMapper,
    ("transformer", "mamba"):       TransformerToMambaMapper,
    ("transformer", "moe"):         DenseToMoEMapper,
    ("llama", "custom"):            LlamaToCustomMapper,
}
```

**Base Mapper Interface:**
```python
class BaseMapper(ABC):
    @abstractmethod
    def map_embedding(self, source: EmbeddingGroup) -> EmbeddingGroup
    
    @abstractmethod
    def map_attention(self, source: AttentionGroup) -> Any
    
    @abstractmethod
    def map_ffn(self, source: FFNGroup) -> FFNGroup
    
    @abstractmethod
    def map_norm(self, source: NormGroup) -> NormGroup
    
    @abstractmethod
    def map_head(self, source: HeadGroup) -> HeadGroup
    
    def map_layer(self, source_layer: Layer) -> Layer:
        # Dispatches to above methods based on layer type
        ...
```

---

### 4.5 Constructor Module

**Responsibility:** Assemble mapped weights into a valid, loadable target model checkpoint.

```python
class ModelConstructor:
    def __init__(self, target_arch: ArchitectureSpec)
    
    def construct(
        self,
        mapped_weights: Dict[str, Tensor],
        fill_strategy: str = "xavier"  # How to initialize unmapped params
    ) -> ModelSnapshot
    
    def save(
        self,
        snapshot: ModelSnapshot,
        output_path: str,
        format: str = "safetensors"
    )
    
    def validate_shapes(self, snapshot: ModelSnapshot) -> bool
```

---

### 4.6 Validator Module

**Responsibility:** Measure how well the converted model preserved the original's capabilities.

```python
class ConversionValidator:
    def perplexity_delta(
        self,
        source_model,
        target_model,
        test_corpus: str = "wikitext2"
    ) -> float   # Lower is better; 0 = identical

    def output_similarity(
        self,
        source_model,
        target_model,
        prompts: List[str]
    ) -> CosineSimilarity

    def representational_similarity(
        self,
        source_model,
        target_model,
        inputs: Tensor
    ) -> RSAScore

    def generate_report(self) -> ValidationReport
```

**Quality Tiers:**
```
Tier 1 (Excellent):   >85% performance retention
Tier 2 (Good):        70–85% performance retention
Tier 3 (Acceptable):  55–70% performance retention
Tier 4 (Degraded):    <55% performance retention → recommend calibration
```

---

## 5. Interface Design

### 5.1 CLI Interface

```bash
# Basic conversion
paradom convert \
  --source meta-llama/Llama-3-8B \
  --target-arch mamba \
  --target-config configs/mamba_1.4b.yaml \
  --output ./output/llama3_as_mamba

# With calibration
paradom convert \
  --source mistralai/Mistral-7B-v0.3 \
  --target-arch custom \
  --target-config configs/my_arch.yaml \
  --calibrate \
  --calibration-data ./data/calibration_sample.jsonl \
  --output ./output/my_sovereign_model

# Validate existing conversion
paradom validate \
  --source ./models/original \
  --converted ./output/my_sovereign_model \
  --benchmark perplexity,task_bench \
  --report ./reports/conversion_quality.json

# List supported conversions
paradom list-mappings

# Inspect model architecture
paradom inspect meta-llama/Llama-3-70B
```

### 5.2 Python API

```python
from paradom import Paradom, ArchitectureSpec, ConversionConfig

# Define target architecture
target = ArchitectureSpec.from_yaml("configs/my_arch.yaml")

# Configure conversion
config = ConversionConfig(
    calibrate=True,
    calibration_samples=500,
    fill_strategy="xavier",
    output_format="safetensors",
    validate_after=True
)

# Run conversion
engine = Paradom()
result = engine.convert(
    source="meta-llama/Llama-3-8B",
    target=target,
    config=config,
    output_path="./output/my_model"
)

# Check quality
print(f"Conversion quality: {result.quality_tier}")
print(f"Perplexity delta: {result.perplexity_delta:.2f}")
print(f"Output similarity: {result.output_similarity:.3f}")

# Load converted model
model = engine.load("./output/my_model")
output = model.generate("Hello, world!")
```

---

## 6. Storage & State Management

### 6.1 Intermediate State

During conversion, AWFE stores:
```
/tmp/paradom_workspace/
├── source_snapshot.pkl       # Parsed source model
├── decomposed/
│   ├── layer_0_attn.npz     # SVD factors per layer
│   ├── layer_0_ffn.npz
│   └── ...
├── mapped/
│   ├── layer_0_mapped.pt    # Mapped weights per layer
│   └── ...
└── conversion_log.json       # Full audit trail
```

### 6.2 Output Format

```
./output/my_model/
├── config.json               # Target architecture config
├── model.safetensors         # Converted weights (or shards)
├── tokenizer/                # Tokenizer (copied from source)
├── conversion_report.json    # Quality metrics
├── paradom_metadata.json        # Conversion provenance
```

---

## 7. Deployment Architecture

### 7.1 Single Machine (Development)

```
┌────────────────────────────────┐
│         Developer Machine      │
│                                │
│  paradom convert ...           │
│       ↓                        │
│  [Paradom Process]             │
│   - CPU: SVD computations      │
│   - GPU: Forward passes (opt.) │
│   - RAM: ~2× model size        │
│       ↓                        │
│  ./output/converted_model/     │
└────────────────────────────────┘

Minimum Requirements:
- RAM: 2× source model size (e.g., 16GB for 7B model in fp16)
- CPU: 8+ cores recommended (parallelized SVD)
- GPU: Optional (speeds up calibration 10×)
- Storage: 3× model size during conversion
```

### 7.2 Cloud / HPC Deployment

```
┌─────────────────────────────────────────────────────┐
│                   Cloud Environment                 │
│                                                     │
│  ┌────────────┐    ┌───────────────────────────┐   │
│  │ Orchestrator│    │    Worker Pool             │   │
│  │            │───▶│                            │   │
│  │ - Job queue│    │  Worker 1: Layers 0–8      │   │
│  │ - Progress │    │  Worker 2: Layers 9–16     │   │
│  │ - Results  │    │  Worker 3: Layers 17–24    │   │
│  └────────────┘    │  Worker N: Layers ...      │   │
│                    └───────────────────────────┘    │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │              Shared Storage                   │  │
│  │  Source weights │ Intermediate │ Output       │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘

Layer-parallel conversion reduces wall time by N_workers
```

### 7.3 Memory-Efficient Streaming Mode

For models too large to fit in RAM:
```python
# Stream conversion layer by layer
engine = Paradom(mode="streaming")
engine.convert_streaming(
    source="meta-llama/Llama-3-70B",
    target=target,
    output="./output/",
    max_ram_gb=24  # Paradom manages memory within this budget
)
```
