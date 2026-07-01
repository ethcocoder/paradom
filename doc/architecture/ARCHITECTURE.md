# System Architecture: Paradom Framework

**Document:** PARADOM-ARCH-001  
**Version:** 1.0.0  
**Date:** 2026-06-30

---

## 1. High-Level Strategy: Spectral Translation

Unlike traditional converters that use rigid mapping, Paradom implements a **Spectral Translation Pipeline**. It treats the model as a collection of functional derivations ($3 = 4 - 1$).

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PARADOM SWAPPER ENGINE                       │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │  SOURCE      │    │   SPECTRAL   │    │  TARGET                │ │
│  │  DERIVATION  │───▶│   MIRROR     │───▶│  DERIVATION            │ │
│  │             │    │              │    │                        │ │
│  │ weights as  │    │ 1. Decompose │    │  Redressed weights     │ │
│  │ "products"  │    │ 2. Align     │    │  as new products       │ │
│  │             │    │ 3. Swap      │    │                        │ │
│  └─────────────┘    └──────────────┘    └────────────────────────┘ │
│           ▲                │                      │                 │
│           └────────────────┴──────────────────────┘                 │
│                 STREAMING MEMORY-MAP PIPELINE                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components: The Swapper Registry

### 2.1 The Swapper Pipeline
The engine is composed of specialized **Swappers** that handle localized functional equivalents:

- **AttentionSwapper**: Maps the $QK^T$ product and the value spectrum.
- **FFNSwapper**: Maps the non-linear derivation of gated activations.
- **EmbeddingSwapper**: Preserves the relational topology of the token space.
- **HeadSwapper**: Aligns the unembedding spectral energy.

### 2.2 Component Map (paradom/)

```
paradom/
├── core/
│   ├── engine.py           # Swapping orchestrator (Streaming logic)
│   ├── snapshot.py         # Memory-mapped model interface
│   ├── decomposer.py       # Spectral decomposition (SVD/Eigen)
│   └── factory.py          # Weight derivation & construction
│
├── swappers/
│   ├── registry.py         # Swapper lookup & dispatch
│   ├── attention.py        # Spectral mirroring for attention
│   ├── ffn.py              # Modular swapping for gated layers
│   └── embeddings.py       # Dimensional projection & topology
│
├── architectures/
│   ├── spec.py             # Functional derivation schema (YAML)
│   ├── registry.py         # Known architecture signatures
│   └── validator.py        # Shape & logic verification
│
├── calibration/
│   ├── spectral_match.py   # Energy-based zero-shot alignment
│   └── activation_sync.py  # Layer-wise statistical correction
│
└── cli/
    └── paradom_cli.py      # Swapping CLI interface
```

---

## 3. Data Flow: The Streaming Swapper

### 3.1 Spectral Transformation Flow

```
1. DISCOVERY:  Identify source matrix "products" via Architecture Signature.
2. MMAP:       Memory-map target layer weights from disk.
3. DECOMPOSE:  Extract the spectral energy (Eigenvalues/Vectors) from source.
4. DERIVE:     Apply $3=4-1$ logic to find target derivation.
5. SWAP:       Overwrite target mmap space with redressed weights.
6. FLUSH:      Commit to disk and move to next layer group.
```

---

## 4. Technical Specifications

### 4.1 Low-Resource Memory-Mapping
Paradom is architected to run on consumer hardware by utilizing **Lazy Weight Swapping**.

```python
class StreamingSwapper:
    def process(self, source_path, target_spec):
        # 1. Open target as a writable mmap file
        with MmapFile(target_path, "w") as target_file:
            # 2. Iterate through "products" (layers) 
            for product in self.registry.identify_products(source_path):
                # Only load ONE product into RAM at a time
                source_weights = self.loader.load_product(product)
                
                # Spectral Decomposition
                spectrum = self.decomposer.get_spectrum(source_weights)
                
                # Derivation Swapping
                new_weights = self.swapper.derive(spectrum, target_spec)
                
                # Direct Write to Disk
                target_file.write_product(product, new_weights)
```

---

## 5. Performance Targets

| Metric | Threshold | Note |
|---|---|---|
| RAM Usage (7B) | <12GB | Enabling 16GB laptops |
| RAM Usage (70B) | <32GB | Enabling enthusiast desktops |
| CPU Scaling | Linear | Parallel SVD across cores |
| I/O Bonded | Yes | Performance limited by disk speed (NVMe recommended) |

---

**Architecture Conclusion:**
Paradom moves away from the monolithic "calculate overall" approach. By swapping specific **functional metrics** and matrix **products** in a streaming pipeline, we achieve high-precision results with minimal resource overhead.
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
