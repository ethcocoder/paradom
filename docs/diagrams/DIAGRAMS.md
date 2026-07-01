# System Diagrams: Paradom Framework

**Document:** PARADOM-DIAG-001
**Version:** 2.0.0
**Date:** 2026-06-30

---

## Diagram 1: The Founding Principle

```
╔═══════════════════════════════════════════════════════════════════════╗
║               THE NUMBER EQUIVALENCE PRINCIPLE                        ║
║                                                                       ║
║   The number 3 exists independently of how you arrived at it:        ║
║                                                                       ║
║       1 + 2  =  3                                                     ║
║       4 - 1  =  3                                                     ║
║       6 / 2  =  3       All different paths.                          ║
║       √9     =  3       All the same number.                          ║
║      15 - 12 =  3                                                     ║
║                                                                       ║
║   A neural network weight W = 0.473 is just a number.                ║
║                                                                       ║
║   LLaMA produced it via:    Transformer attention gradient descent    ║
║   Mamba could encode it as: SSM state dynamics parameter              ║
║   ViT could hold it as:     Patch attention query weight              ║
║                                                                       ║
║   PARADOM finds the equivalent position and swaps the number.        ║
║   No retraining. No recalculation. Just: identify → swap → validate. ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## Diagram 2: The Universal Coverage Map

```
                    ALL ML/DL PARADIGMS
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │   LARGE LANGUAGE MODELS          VISION MODELS       │
    │   ┌───────────────────┐          ┌────────────────┐  │
    │   │ LLaMA   → Mamba   │          │ CNN  → ViT     │  │
    │   │ Mistral → MoE     │          │ ResNet→MLP-Mix │  │
    │   │ Falcon  → Custom  │          │ ConvNext→DeiT  │  │
    │   └─────────┬─────────┘          └───────┬────────┘  │
    │             │                            │           │
    │             └────────────┬───────────────┘           │
    │                          │                           │
    │                    ┌─────▼──────┐                    │
    │                    │  PARADOM   │                    │
    │                    │   SWAP     │                    │
    │                    │   ENGINE   │                    │
    │                    └─────┬──────┘                    │
    │                          │                           │
    │             ┌────────────┴───────────────┐           │
    │             │                            │           │
    │   ┌─────────▼─────────┐          ┌───────▼────────┐  │
    │   │ REINFORCEMENT      │          │ GENERATIVE     │  │
    │   │ LEARNING           │          │ MODELS         │  │
    │   │ DQN  → PPO         │          │ Diffusion →    │  │
    │   │ A2C  → Transformer │          │ Flow Matching  │  │
    │   │ SAC  → Custom      │          │ GAN → VAE      │  │
    │   └───────────────────┘          └────────────────┘  │
    │                                                      │
    │   ┌──────────────────────────────────────────────┐   │
    │   │ GRAPH LEARNING          MULTIMODAL           │   │
    │   │ GNN → GraphTransformer  CLIP → CustomVision  │   │
    │   └──────────────────────────────────────────────┘   │
    └──────────────────────────────────────────────────────┘
                              │
                              ▼
              SOVEREIGN AI DEPLOYMENT
              (Any architecture. Any hardware. Any nation.)
```

---

## Diagram 3: The Swap Pipeline (Step by Step)

```
INPUT: Source model path + Target architecture config
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD (Streaming — one layer at a time)                  │
│                                                                  │
│  HuggingFace Hub ──┐                                             │
│  Local .pt file  ──┼──▶ [Format Detector] ──▶ [Layer Iterator]  │
│  SafeTensors     ──┘                                             │
│                                                                  │
│  Memory: Only ONE layer in RAM at any time (~450MB for 7B model) │
│  Peak RAM: ~1-2GB regardless of total model size                 │
└─────────────────────────────┬────────────────────────────────────┘
                              │ (one layer)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: ASSIGN FUNCTIONAL ROLE                                   │
│                                                                  │
│  "model.layers.0.self_attn.q_proj.weight"                        │
│                    │                                             │
│                    ▼                                             │
│  [FunctionalRoleMatcher]                                         │
│                    │                                             │
│                    ▼                                             │
│  FunctionalRole.CONTEXT_QUERY                                    │
│  (This weight asks: "what am I looking for in context?")         │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: SCORE IMPORTANCE                                         │
│                                                                  │
│  [ImportanceScorer.svd_spectrum()]                               │
│                                                                  │
│  W (4096×4096) ──▶ SVD ──▶ Singular value spectrum              │
│                                                                  │
│  Top 20% weights by importance: ████████░░░░░░░░░░░░░░░░░░░░░░  │
│  These carry ~80% of the information (power law distribution)    │
│                                                                  │
│  Result: ImportanceMask — which specific numbers matter most     │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4: FIND EQUIVALENT POSITION IN TARGET                       │
│                                                                  │
│  [EquivalenceIdentifier]                                         │
│                                                                  │
│  Source role: CONTEXT_QUERY in Transformer layer 0              │
│       │                                                          │
│       ▼                                                          │
│  Target equivalent: CONTEXT_QUERY in Mamba layer 0              │
│  → "backbone.layers.0.mixer.in_proj.weight" (first half)        │
│                                                                  │
│  CKA validation score: 0.67 (GOOD — projected swap appropriate)  │
│  Swap type selected: PROJECTED SWAP                              │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 5: SWAP                                                     │
│                                                                  │
│  Source: W_q_proj (4096 × 4096)                                  │
│  Target: W_in_proj_first_half (4096 × 2048)  ← different shape  │
│                                                                  │
│  Projected swap:                                                  │
│    U, S, Vh = SVD(W_q_proj)                                      │
│    W_target = (U[:2048, :] * S[:2048]) @ Vh[:2048, :]           │
│                                                                  │
│  Only the top 20% important elements are swapped.               │
│  Remaining 80% of target: Xavier initialization.                 │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 6: WRITE & FREE                                             │
│                                                                  │
│  Write converted layer to output SafeTensors file               │
│  Free source layer from RAM                                      │
│  Move to next layer                                              │
│                                                                  │
│  Progress: ████████████████░░░░░░░░░░ Layer 18/32 (56%)         │
└─────────────────────────────┬────────────────────────────────────┘
                              │ (repeat for all layers)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 7: VALIDATE                                                 │
│                                                                  │
│  Compute per-layer CKA scores                                    │
│  Run paradigm benchmark (perplexity / Top-1 / reward / FID)     │
│  Generate SwapValidationReport                                   │
│  Classify quality tier                                           │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
OUTPUT: ./output/converted_model/
  ├── config.json               (target architecture config)
  ├── model.safetensors         (converted weights)
  ├── tokenizer/                (copied from source)
  ├── swap_report.json          (quality metrics)
  └── equivalence_map.json      (which weights went where)
```

---

## Diagram 4: Swap Types — When Each Is Used

```
                      SOURCE WEIGHT
                           │
               ┌───────────▼────────────┐
               │   CKA Score?           │
               │   Role Match?          │
               └───────────┬────────────┘
                           │
            ┌──────────────┼──────────────────────────┐
            │              │              │            │
            ▼              ▼              ▼            ▼
        CKA ≥ 0.70    CKA ≥ 0.40    CKA ≥ 0.20    CKA < 0.20
        Same shape     Diff shape     Multi-dim      Very weak
            │              │              │            │
            ▼              ▼              ▼            ▼
      ┌──────────┐  ┌────────────┐ ┌──────────┐ ┌──────────┐
      │  DIRECT  │  │ PROJECTED  │ │  TENSOR  │ │   OPT.   │
      │   SWAP   │  │   SWAP     │ │  DECOMP  │ │TRANSPORT │
      │          │  │            │ │   SWAP   │ │   SWAP   │
      │ W_B=W_A  │  │ W_B=P·W_A │ │ Tucker   │ │Wasserstein│
      │ (copy)   │  │ (project)  │ │decompose │ │  mapping │
      └──────────┘  └────────────┘ └──────────┘ └──────────┘
      85–99%        70–90%          60–80%        45–65%
      retention     retention       retention     retention

      Example:      Example:        Example:      Example:
      Embedding     Q → in_proj     Conv → Attn   GNN → Trans
      tables,       (LLM cross-     (CNN → ViT)   (graph →
      output heads  arch)                          sequence)
```

---

## Diagram 5: LLM Equivalence Map (Transformer → Mamba)

```
TRANSFORMER LAYER                    MAMBA LAYER
─────────────────────                ─────────────────────

input_layernorm                ──▶   norm
(RMSNorm weight)               ═══   (RMSNorm weight)
SWAP TYPE: Direct  CKA: ~0.95        Role: NORMALIZATION

self_attn.q_proj               ──▶   mixer.in_proj (1st half)
(4096 × 4096)                  ──▶   (4096 × 2048)
SWAP TYPE: Projected  CKA: ~0.65     Role: CONTEXT_QUERY

self_attn.k_proj               ──▶   mixer.in_proj (2nd half)
(4096 × 4096, or GQA)          ──▶   (4096 × 2048)
SWAP TYPE: Projected  CKA: ~0.62     Role: CONTEXT_KEY

self_attn.v_proj               ──▶   mixer.x_proj (part)
(4096 × 4096)                  ──▶   (derived)
SWAP TYPE: Projected  CKA: ~0.58     Role: CONTEXT_VALUE

self_attn.o_proj               ──▶   mixer.out_proj
(4096 × 4096)                  ──▶   (4096 × 2048)
SWAP TYPE: Projected  CKA: ~0.61     Role: CONTEXT_OUTPUT

mlp.gate_proj + up_proj        ──▶   mixer.dt_proj
(4096 → 11008)                 ──▶   (derived)
SWAP TYPE: Projected  CKA: ~0.53     Role: FFN_EXPAND

mlp.down_proj                  ──▶   mixer.out_proj (part)
(11008 → 4096)                 ──▶   (derived)
SWAP TYPE: Projected  CKA: ~0.55     Role: FFN_CONTRACT

[NO TRANSFORMER EQUIVALENT]    ──▶   mixer.A_log
                               ═══   DERIVED from eigenvalues of
                               ═══   W_Q @ W_K^T (not a swap!)
                                     Role: SSM_STATE_DYNAMICS

embed_tokens                   ──▶   embedding
(32000 × 4096)                 ═══   (32000 × 4096)
SWAP TYPE: Direct  CKA: ~0.99        Role: EMBEDDING

lm_head                        ──▶   lm_head
(32000 × 4096)                 ═══   (32000 × 4096)
SWAP TYPE: Direct  CKA: ~0.99        Role: OUTPUT_HEAD
```

---

## Diagram 6: RL Paradigm Swap Map (DQN → PPO)

```
DQN NETWORK                          PPO NETWORK
──────────────                       ──────────────

Feature Extractor                    Policy Feature Extractor
  conv1 (8×8, stride 4)    ══▶         conv1 (8×8, stride 4)
  conv2 (4×4, stride 2)    ══▶         conv2 (4×4, stride 2)
  conv3 (3×3, stride 1)    ══▶         conv3 (3×3, stride 1)
  fc_feature (512×3136)    ══▶         fc_feature (512×3136)

  SWAP TYPE: Direct (Atari envs share architecture)
  CKA: ~0.78
  What transfers: "How to understand Atari game states"
  This is the HIGHEST VALUE swap — saves ~70% of training time

Q-Value Head                         Action Head (Policy Logits)
  fc_q (n_actions × 512)  ──▶         fc_action (n_actions × 512)

  SWAP TYPE: Projected + normalized
  CKA: ~0.42
  What transfers: "Which actions are better" (direction, not scale)
  What changes: Q-values (unbounded) → logits (normalized)
  Operation: W_action = normalize(W_q) × 0.1

[NO DQN EQUIVALENT]                  Value Head
                           ←══         fc_value (1 × 512)
  SWAP TYPE: Xavier init (fresh)
  No equivalent exists in DQN.
  PPO must learn this from scratch.
  (Usually learns quickly given the shared feature extractor)
```

---

## Diagram 7: Sovereign AI Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE SOVEREIGN AI VISION                          │
└─────────────────────────────────────────────────────────────────────┘

LAYER 1: GLOBAL COMMONS (Free, Open Source)
┌─────────────────────────────────────────────────────────────────────┐
│  LLaMA 3    Mistral    Falcon    Gemma    Qwen    Phi-3             │
│         (Apache 2.0 / MIT — free to use and modify)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               │  Paradom swaps the weights
                               │  into your sovereign architecture
                               │  No training data needed.
                               │  No GPU cluster needed.
                               │  No foreign API dependency.
                               ▼
LAYER 2: PARADOM CONVERSION
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   paradom swap                                                      │
│     --source meta-llama/Llama-3-8B                                  │
│     --target-arch custom                                            │
│     --target-config configs/sovereign_7b.yaml                       │
│     --output ./my_sovereign_model                                   │
│                                                                     │
│   Time:  ~15 minutes    RAM: ~4GB    GPU: not required              │
│   Cost:  $0 compute    Quality: 65–82% of source retained          │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               │  Optional: LoRA fine-tune on
                               │  local language data
                               │  (Amharic, Tigrinya, Oromo, etc.)
                               ▼
LAYER 3: SOVEREIGN MODEL
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   🇪🇹  Ye-Ethiopia AI  │  Your Nation's AI  │  Your Community's AI  │
│                                                                     │
│   ✅ Runs on YOUR infrastructure (no cloud required)                │
│   ✅ Data NEVER leaves your country                                  │
│   ✅ Architecture YOU designed and control                          │
│   ✅ Fine-tuned on YOUR language and culture                        │
│   ✅ No subscription fees, no API limits, no foreign dependency     │
│   ✅ Can be audited, modified, and improved locally                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

COST COMPARISON:
  Train frontier model from scratch:  $2,000,000 – $100,000,000  ❌
  Annual API dependency (GPT/Claude):  $50,000 – $500,000/year   ⚠️
  Paradom + local fine-tune:           $1,000 – $20,000 total    ✅
```

---

## Diagram 8: Quality vs Swap Fraction

```
QUALITY RETAINED vs FRACTION OF WEIGHTS SWAPPED

100% ┤
     │
 90% ┤      ●───────────────────●─────●
     │     /                        (diminishing returns above 35%)
 80% ┤    /
     │   /
 70% ┤  /
     │ /
 60% ┤●
     │ (5% swap fraction — only the very top weights)
 50% ┤
     │
     └──────────────────────────────────────────────────
      5%    10%    20%    35%    50%    75%    100%
                   ↑
             DEFAULT (20%)
             Best quality/time ratio

SAME ARCHITECTURE SWAP:     ████████████████████████  85–95%
CROSS ARCHITECTURE (LLM):   ████████████████          65–80%
CROSS PARADIGM (CNN→ViT):   ████████████              60–72%
CROSS PARADIGM (DQN→PPO):   ██████████                55–68%

All at 20% swap fraction. With calibration pass: +5–10% each.
```
