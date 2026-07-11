# Paradom Development Roadmap

## Phase 1: Number Equivalence (ARCHIVED ✅)
*Proof of concept successfully demonstrated >100% relative retention (19.04 ppl vs 22.46 baseline).*

## Phase 2: Swap Engine — LLM Paradigm (ARCHIVED ✅)

### Month 4: Infrastructure & 7B Scale
- [x] Implement Optimized SVD (Randomized)
- [x] Implement Metadata Streaming in Loader
- [x] Implement `StreamingSwapper` for 70B support
- [x] Create LLaMA 3 8B → Mamba 7B mapping spec
- [x] Benchmark Randomized SVD vs Exact SVD speed

### Month 5: MoE & Generic Mappers
- [x] Implement `DenseToMoEMapper` (FFN → Experts)
- [x] Implement `TransformerToTransformerMapper` (Arch-morphing)
- [x] Expand `MAPPING_REGISTRY`

### Month 6: Public Benchmarks
- [x] Validation suite for HellaSwag & ARC
- [x] Technical Report: Scaling Weight Force Equivalence
