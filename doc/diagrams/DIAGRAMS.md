# System Diagrams: Paradom Framework

**Document:** PARADOM-DIAG-001  
**Version:** 1.0.0

---

## Diagram 1: High-Level Concept

```
╔══════════════════════════════════════════════════════════════════╗
║                    THE PARADOM CONCEPT                           ║
║                                                                  ║
║   "Same knowledge, different mathematical representation"        ║
║                                                                  ║
║   1 + 2 = 3    ←────────────────────────────────→   4 - 1 = 3  ║
║   (LLaMA arch)     Different paths, same result    (Mamba arch) ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

         Hugging Face / Kaggle
         ┌─────────────────────────────────────────────┐
         │  LLaMA 3   Mistral   Falcon   Gemma  Qwen   │
         │  (Free, open-source, pre-trained weights)   │
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
         │             PARADOM ENGINE                  │
         │                                             │
         │  Parse → Decompose → Map → Construct        │
         │                                             │
         │  (Mathematical translation of weights)      │
         └──────────────────────┬──────────────────────┘
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
           ┌──────────┐  ┌──────────┐  ┌──────────┐
           │  Custom  │  │  Mamba   │  │   MoE    │
           │  Arch    │  │   SSM    │  │  Expert  │
           │          │  │          │  │          │
           │Sovereign │  │Efficient │  │ Scaling  │
           │   AI     │  │ Inference│  │ Arch     │
           └──────────┘  └──────────┘  └──────────┘
```

---

## Diagram 2: Conversion Pipeline (Detailed)

```
INPUT
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: LOAD                                                   │
│                                                                 │
│  HF Hub ──▶ [Downloader] ──▶ [Format Detector]                 │
│  Local  ──▶ [File Reader]         │                             │
│  GGUF   ──▶ [GGUF Parser]         ▼                             │
│                            [ModelSnapshot]                      │
│                            {architecture, config, layers}       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: PARSE                                                  │
│                                                                 │
│  [ModelSnapshot]                                                │
│        │                                                        │
│        ▼                                                        │
│  [ArchitectureParser]                                           │
│        │                                                        │
│        ├──▶ EmbeddingGroup  (token + position embeddings)       │
│        ├──▶ Layer 0:                                            │
│        │     ├── AttentionGroup  (W_Q, W_K, W_V, W_O)          │
│        │     ├── FFNGroup        (W_gate, W_up, W_down)         │
│        │     └── NormGroup       (weight, bias)                 │
│        ├──▶ Layer 1 ... Layer N                                 │
│        └──▶ HeadGroup     (output projection)                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: DECOMPOSE                                              │
│                                                                 │
│  For each AttentionGroup:                                       │
│    W_Q, W_K ──▶ [eigendecompose] ──▶ {eigenvalues, eigenvecs}  │
│    W_V, W_O ──▶ [SVD]            ──▶ {U, S, Vh}                │
│                                                                 │
│  For each FFNGroup:                                             │
│    W_up     ──▶ [SVD]            ──▶ {U, S, Vh}                │
│                                                                 │
│  For EmbeddingGroup:                                            │
│    E        ──▶ [PCA basis]      ──▶ {principal_components}     │
│                                                                 │
│  Result: [DecomposedModel] {layer factorizations}               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: MAP                                                    │
│                                                                 │
│  [MappingRegistry].get(source_arch, target_arch)               │
│        │                                                        │
│        ▼                                                        │
│  [Selected Mapper]                                              │
│        │                                                        │
│        ├── map_embedding()  ──▶ target embedding weights        │
│        ├── map_attention()  ──▶ target attention weights        │
│        │       (or SSM weights, or MoE weights)                 │
│        ├── map_ffn()        ──▶ target FFN weights              │
│        ├── map_norm()       ──▶ target norm weights             │
│        └── map_head()       ──▶ target output weights           │
│                                                                 │
│  Result: [MappedWeights] {target-shaped tensors}                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: CONSTRUCT                                              │
│                                                                 │
│  [ModelConstructor(target_spec)]                                │
│        │                                                        │
│        ├── Assemble target weight dict                          │
│        ├── Verify all required tensors present                  │
│        ├── Initialize missing tensors (Xavier)                  │
│        ├── Convert dtype (fp32 → bf16)                          │
│        └── Write to SafeTensors                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 6: CALIBRATE (Optional)                                   │
│                                                                 │
│  Zero-shot:                                                     │
│    Synthetic inputs ──▶ Activation collection                   │
│    Src activations ──▶ Affine correction ──▶ Apply to weights  │
│                                                                 │
│  Few-shot:                                                      │
│    Calibration data ──▶ CMA-ES optimization ──▶ Better weights  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 7: VALIDATE                                               │
│                                                                 │
│  Perplexity test ──▶ perplexity_ratio                          │
│  Output similarity ──▶ cosine_similarity                        │
│  Task benchmarks ──▶ {hellaswag, arc, mmlu}                    │
│  Quality tier ──▶ "excellent" | "good" | "acceptable"          │
│                                                                 │
│  ValidationReport saved to ./output/conversion_report.json     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
OUTPUT: ./output/<model>/
  ├── config.json
  ├── model.safetensors
  ├── tokenizer/
  ├── conversion_report.json
  └── paradom_metadata.json
```

---

## Diagram 3: The Attention → SSM Mathematical Bridge

```
TRANSFORMER ATTENTION                    MAMBA SSM
─────────────────────                    ─────────

Input: x ∈ ℝ^(T × d)                    Input: x ∈ ℝ^(T × d)
                                         
Q = x @ W_Q    ∈ ℝ^(T × d_head)        
K = x @ W_K    ∈ ℝ^(T × d_head)        State: h_t ∈ ℝ^N
V = x @ W_V    ∈ ℝ^(T × d_head)        
                                         
Attn = softmax(QK^T / √d) @ V           h_t = A·h_{t-1} + B·x_t
Output = Attn @ W_O                      y_t = C·h_t
         
Key pattern matrix:                      
P = W_Q @ W_K^T ∈ ℝ^(d × d)           
                                         
         PARADOM BRIDGE
         ──────────
                                         
Eigendecompose P:                        
P = V · Λ · V^T                         
                                         
         │                              
         ▼                              
Top N eigenvalues (λ₁,...,λ_N)   ────▶  A_log = log(|λᵢ|)    (State dynamics)
Top N eigenvectors V_N           ────▶  B = V_N^T @ W_V       (Input proj)
                                 ────▶  C = W_O @ V_N          (Output proj)
                                         
RESULT: SSM that approximates the attention's                    
context-selection behavior.              
Quality depends on how well N eigenvalues                        
capture the attention pattern.           
```

---

## Diagram 4: Quality Tiers Visual

```
CONVERSION QUALITY SPECTRUM
────────────────────────────────────────────────────────────────

  Source Model Performance = 100%
  │
  │  ████████████████████████████████████████████████  100%
  │                                                        ▲
  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          ║
  │                                              85%    EXCELLENT
  │  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒             ║
  │                                        70%       GOOD
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░                   ║
  │                                55%           ACCEPTABLE
  │  ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪                           ║
  │                  <55%                     DEGRADED
  │                                         (calibrate!)
  0%

EXPECTED PARADOM QUALITY BY CONVERSION TYPE:
  Same-arch (Phase 1):          ████████████████████  85–95% EXCELLENT
  Cross-arch + calibration:     ████████████████      70–85% GOOD
  Transformer → Mamba:          ████████████          60–75% GOOD/ACCEPTABLE
  Transformer → Mamba (raw):    ████████              50–65% ACCEPTABLE
```

---

## Diagram 5: Sovereignty Architecture

```
THE SOVEREIGN AI STACK (Powered by Paradom)
──────────────────────────────────────────

  GLOBAL COMMONS (Open Source Weights)
  ┌──────────────────────────────────────────────────────────┐
  │   LLaMA 3    Mistral    Falcon    Gemma    Qwen    ...   │
  │   (Apache 2.0 / permissive licenses)                     │
  └──────────────────────────┬───────────────────────────────┘
                             │  Paradom converts weights
                             ▼
  SOVEREIGN CUSTOMIZATION LAYER
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │   Paradom Engine                                         │
  │   ├── Convert to sovereign architecture                  │
  │   ├── Optimize for local hardware                        │
  │   └── Preserve 65-85% of source intelligence            │
  │                                                          │
  └──────────────────────────┬───────────────────────────────┘
                             │  Optional: LoRA fine-tune
                             ▼             on local data
  SOVEREIGN MODEL
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │   🇪🇹 Ye-Ethiopia AI / የኢትዮጵያ AI                       │
  │   (or any nation's sovereign model)                      │
  │                                                          │
  │   • Runs on sovereign infrastructure                     │
  │   • No dependency on foreign APIs                        │
  │   • Customized for local languages & culture             │
  │   • Data stays within the country                        │
  │   • Architecture controlled by the nation               │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

TOTAL COST COMPARISON:
  Train from scratch:   $2M–$100M     ❌ Inaccessible
  Fine-tune only:       $10K–$500K    ⚠️  Dependency on source arch
  Paradom + fine-tune:  $1K–$50K      ✅ Affordable, fully sovereign
```

---

## Diagram 6: Module Dependency Graph

```
                    ┌─────────────┐
                    │    CLI      │
                    │(paradom_cli)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Python API │
                    │(paradom_api)│
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │      Paradom Core       │
              │   (pipeline.py)         │
              └─┬──────┬───────┬────────┘
                │      │       │
         ┌──────▼┐  ┌──▼────┐ ┌▼──────────┐
         │Loader │  │Parser │ │Decomposer │
         └───────┘  └───────┘ └──────┬────┘
                                     │
                              ┌──────▼──────┐
                              │   Mapper    │
                              │  Registry   │
                              └──┬───┬───┬──┘
                   ┌─────────────┘   │   └──────────────┐
            ┌──────▼──────┐   ┌──────▼──────┐   ┌───────▼──────┐
            │  T2T Mapper │   │  T2M Mapper │   │  D2E Mapper  │
            │(Trans→Trans)│   │(Trans→Mamba)│   │(Dense→MoE)  │
            └─────────────┘   └─────────────┘   └──────────────┘
                                     │
                              ┌──────▼──────┐
                              │Constructor  │
                              └──────┬──────┘
                                     │
                     ┌───────────────┼──────────────────┐
              ┌──────▼──────┐ ┌──────▼──────┐  ┌───────▼──────┐
              │  Calibrator │ │  Validator  │  │  Benchmarks  │
              └─────────────┘ └─────────────┘  └──────────────┘
```
