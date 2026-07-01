# Phase 2: Core Engine Development

**Document:** PARADOM-PHASE-002  
**Duration:** Week 2  
**Prerequisites:** Phase 1 complete, ≥70% quality retention demonstrated  
**Goal:** Build and validate the first true cross-architecture conversion (Transformer → Mamba SSM).

---

## Phase 2 Objective

> Implement and validate the TransformerToMambaMapper — the first genuine cross-architecture weight transfer in Paradom. This is the scientifically novel contribution.

---

## Success Criteria

- [ ] TransformerToMambaMapper implemented and tested
- [ ] LLaMA 7B → Mamba 1.4B conversion achieves ≥50% perplexity retention
- [ ] DenseToMoEMapper implemented
- [ ] Streaming mode implemented (handles 70B models with limited RAM)
- [ ] Mapping registry with 4+ supported conversion paths
- [ ] Performance benchmark: 7B conversion in <20 minutes

---

## Week 2: Transformer → Mamba Mapping

### Core Implementation

The attention → SSM mapping is the most mathematically involved step.

**Research Spike (Week 1):**
- Implement toy version on a 2-layer transformer and 2-layer Mamba
- Run 1000 random inputs; compare output distributions
- Measure baseline transfer quality before any optimization
- Document findings in `research/attn_ssm_mapping_experiments.md`

**Implementation (Week 2–4):**

```python
class TransformerToMambaMapper(BaseMapper):
    def map_attention(self, source: AttentionGroup) -> MambaBlock:
        # STAGE 1: Eigendecomposition of attention pattern
        W_QK = source.W_Q @ source.W_K.T
        eigenvalues, eigenvectors = torch.linalg.eigh(W_QK)
        
        # STAGE 2: Select top N state dimensions
        # N is the Mamba state dimension (typically 16-64)
        N = self.target_config.state_dim
        top_idx = torch.argsort(eigenvalues.abs(), descending=True)[:N]
        top_eigenvalues = eigenvalues[top_idx]
        top_eigenvectors = eigenvectors[:, top_idx]
        
        # STAGE 3: Map to SSM parameters
        A_log = self._eigenvalues_to_A_log(top_eigenvalues)
        B = top_eigenvectors.T @ source.W_V[:self.target_config.d_inner, :]
        C = source.W_O[:, :self.target_config.d_inner] @ top_eigenvectors
        
        # STAGE 4: Input/output projections
        in_proj = self._build_in_proj(source.W_V, source.W_O)
        out_proj = self._build_out_proj(source.W_O)
        
        return MambaBlock(
            A_log=A_log, B=B, C=C,
            in_proj=in_proj, out_proj=out_proj,
            conv1d=self._init_conv1d(),
            x_proj=self._build_x_proj(N),
            dt_proj=self._build_dt_proj(source.hidden_size),
            D=torch.ones(self.target_config.d_inner)
        )
```

**Key Questions to Answer in Phase 2:**
1. Which eigenvalues map best to SSM dynamics? Top-k by magnitude? By sign?
2. How sensitive is quality to the choice of state dimension N?
3. Do the FFN layers transfer independently of the attention transfer quality?

---

## Week 2: Streaming Mode & MoE Mapping

### Streaming Conversion (Week 5–6)

Critical for 70B+ models:

```python
class StreamingConverter:
    """Converts models layer by layer to stay within RAM budget."""
    
    def convert(
        self,
        source_path: str,
        target_spec: ArchitectureSpec,
        output_path: str,
        max_ram_gb: float = 24
    ):
        layer_size_estimate = self._estimate_layer_size(source_path)
        
        with SafeTensorsWriter(output_path) as writer:
            for layer_idx in range(self.n_layers):
                # Load ONE layer at a time
                layer_weights = self._load_layer(source_path, layer_idx)
                
                # Convert
                mapped = self.mapper.map_layer(layer_weights)
                
                # Write to output immediately
                writer.write_layer(layer_idx, mapped)
                
                # Free memory
                del layer_weights, mapped
                gc.collect()
                
                self._log_progress(layer_idx)
```

### DenseToMoEMapper (Week 7–8)

- Implement SVD-based expert initialization (see SPECIFICATIONS.md §3.3)
- Validate: converted MoE has same forward pass complexity as target
- Benchmark: ensure expert specialization begins from initialization

---

## Week 2: Mapping Registry & Quality Hardening

**Tasks:**
- Implement `MappingRegistry` with all 4+ conversion paths
- Add `awfe list-mappings` CLI command with quality estimates
- Implement per-layer quality diagnostics: identify which layers lose the most quality
- Write comprehensive Phase 2 benchmark report
- Harden error handling: meaningful errors for unsupported conversions
- Update documentation with cross-architecture examples

**Phase 2 Validation Target:**

```
LLaMA 3 8B → Mamba 1.4B:
  Perplexity ratio:     ≤2.0  (target: ≤1.6)
  Output similarity:    ≥0.40 (target: ≥0.55)
  Conversion time:      ≤20 min

LLaMA 3 8B → Same-arch custom:
  Perplexity ratio:     ≤1.15 (unchanged from Phase 1)
  Conversion time:      ≤15 min

LLaMA 3 8B → MoE-7B (custom):
  Perplexity ratio:     ≤1.40
  Conversion time:      ≤25 min
```

---

---

# Phase 3: Validation & Intelligence Preservation

**Document:** PARADOM-PHASE-003  
**Duration:** Week 3  
**Goal:** Maximize quality of converted models through calibration and bias correction.

---

## Phase 3 Objective

> Develop and validate calibration techniques that improve converted model quality without requiring full retraining. Target: cross-architecture conversion quality reaches Tier 2 (≥70% retention) without calibration data.

---

## Success Criteria

- [ ] Zero-shot calibration implemented and improves quality by ≥15% over raw conversion
- [ ] Few-shot calibration implemented (with 1M tokens budget)
- [ ] Transformer → Mamba achieves ≥65% quality with zero-shot calibration
- [ ] Full benchmark suite on lm-evaluation-harness (HellaSwag, ARC, MMLU)
- [ ] Public benchmark results published

---

## Week 3: Zero-Shot Calibration

### Activation Matching

The key insight: even without real data, we can generate synthetic inputs from the model's own embedding space and use them to align activation statistics.

```python
class ZeroShotCalibrator:
    def generate_synthetic_inputs(self, model, n_samples: int) -> Tensor:
        """
        Generates inputs by sampling from the model's learned embedding space.
        These are 'in-distribution' for the source model by definition.
        """
        embedding_matrix = model.get_input_embeddings().weight  # (vocab × d_model)
        
        # Sample sequences using the embedding distribution
        # Strategy 1: Random walks in embedding space
        inputs = []
        for _ in range(n_samples):
            seq = self._embedding_random_walk(embedding_matrix, length=128)
            inputs.append(seq)
        
        return torch.stack(inputs)
    
    def apply_activation_corrections(
        self,
        source_model,
        converted_model,
        synthetic_inputs: Tensor
    ) -> None:
        """Align activation statistics layer by layer."""
        src_hooks = self._register_activation_hooks(source_model)
        tgt_hooks = self._register_activation_hooks(converted_model)
        
        with torch.no_grad():
            for batch in synthetic_inputs.split(8):
                source_model(batch)
                converted_model(batch)
        
        # Apply per-layer affine corrections
        for layer_name in src_hooks.activations:
            self._compute_and_apply_correction(
                source_model, converted_model, layer_name,
                src_hooks.activations[layer_name],
                tgt_hooks.activations[layer_name]
            )
```

---

## Week 3: Few-Shot Calibration & Benchmarking

### Few-Shot Calibration Pipeline

Target: Use ≤1M tokens (a tiny fraction of typical training data) to improve quality.

```bash
# Example: Calibrate with public domain text
awfe calibrate \
  --converted ./output/llama3_as_mamba \
  --data ./data/calibration_sample.jsonl \
  --method few_shot \
  --budget-tokens 1000000 \
  --output ./output/llama3_as_mamba_calibrated
```

### Full Benchmark Protocol

Run every converted model on:
- **WikiText-2 perplexity** — language modeling baseline
- **HellaSwag** — commonsense reasoning
- **ARC-Easy / ARC-Challenge** — knowledge & reasoning
- **MMLU** — broad knowledge evaluation
- **TruthfulQA** — factual accuracy

Publish results on project website to establish public credibility.

---

## Week 3: Quality Hardening & Public Report

**Tasks:**
- Identify and fix top 5 failure modes in conversion quality
- Write `KNOWN_LIMITATIONS.md` — honest about what AWFE can and cannot do
- Write quality prediction model: given source+target arch, predict conversion quality before running
- Prepare Phase 3 technical report for potential academic publication
- Community feedback: share with 3–5 external ML engineers for review

---

---

# Phase 4: Scale & Production

**Document:** PARADOM-PHASE-004  
**Duration:** Week 4  
**Goal:** Make Paradom production-ready, scalable, and accessible to real users.

---

## Phase 4 Objective

> Transform Paradom from a research tool into a production-grade framework that real engineers can use to convert 70B+ models in organizational/cloud settings.

---

## Success Criteria

- [ ] Successfully convert LLaMA 70B model end-to-end in <3 hours
- [ ] REST API server for conversion-as-a-service
- [ ] Web UI for non-technical users
- [ ] Package published to PyPI (`pip install awfe`)
- [ ] Docker image available (`docker pull awfe/awfe:latest`)
- [ ] First external user successfully converts a model using AWFE

---

## Week 4: Scale & Performance

### 70B Model Support

The main challenges at 70B scale:
- **Memory:** 70B fp16 = 140GB weights (exceeds single machine RAM)
- **Time:** SVD on 8192×8192 matrices is slow
- **Stability:** Numerical precision issues at scale

**Solutions:**

```python
# Blocked SVD for large matrices
def blocked_svd(W: Tensor, block_size: int = 2048) -> SVDFactors:
    """Memory-efficient SVD by processing matrix in blocks."""
    n, m = W.shape
    n_blocks = math.ceil(n / block_size)
    
    # Use randomized SVD (much faster for approximate decomposition)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False, driver='gesvda')
    return SVDFactors(U, S, Vh, rank=len(S), energy=1.0)

# Parallel layer processing
class ParallelConverter:
    def convert(self, model, n_workers: int = 8):
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            for layer_idx in range(model.n_layers):
                future = pool.submit(self._convert_layer, model, layer_idx)
                futures.append(future)
            results = [f.result() for f in tqdm(futures)]
```

---

## Week 4: REST API & Web UI

### Conversion API Server

```python
# awfe/server/api.py
from fastapi import FastAPI, BackgroundTasks

app = FastAPI(title="AWFE Conversion API")

@app.post("/conversions")
async def start_conversion(
    request: ConversionRequest,
    background_tasks: BackgroundTasks
):
    job_id = str(uuid4())
    background_tasks.add_task(run_conversion, job_id, request)
    return {"job_id": job_id, "status": "queued"}

@app.get("/conversions/{job_id}")
async def get_status(job_id: str):
    return conversion_store.get(job_id)

@app.get("/conversions/{job_id}/download")
async def download_model(job_id: str):
    return FileResponse(conversion_store.get_output_path(job_id))
```

### Simple Web UI

A minimal web interface so researchers without CLI experience can:
1. Paste a HuggingFace model ID
2. Select or upload a target architecture YAML
3. Click "Convert"
4. Download the result

---

## Week 4: Polish, PyPI Release & First External Users

**Tasks:**
- Full documentation website (Sphinx or MkDocs)
- PyPI release with semantic versioning
- Docker image (`awfe/awfe:latest`)
- Integration tests against 10+ source models
- `pip install awfe && awfe convert ...` must work in 5 minutes from scratch
- Onboard 3–5 beta users and fix friction points
- Write blog post: "Introducing AWFE: Convert Any Open Model to Any Architecture"

---

---

# Phase 5: Open Source & Sovereignty Release

**Document:** PARADOM-PHASE-005  
**Duration:** Week 5+  
**Goal:** Establish Paradom as the global standard for cross-architecture model conversion, with a specific focus on enabling sovereign AI.

---

## Phase 5 Objective

> Launch Paradom publicly, build the community, and demonstrate real-world impact by helping at least one organization build a sovereign AI system using Paradom.

---

## Success Criteria

- [ ] GitHub repository reaches 500+ stars within 3 months of launch
- [ ] At least 1 sovereign AI project (national/regional) uses Paradom
- [ ] Academic paper submitted to peer-reviewed venue
- [ ] 5+ community-contributed architecture definitions
- [ ] Paradom cited in at least 3 external research projects

---

## Sovereign AI Initiative

Paradom's mission includes enabling sovereign AI — the ability for nations and communities to own and control their AI infrastructure.

### Sovereign Deployment Guide

```markdown
# Sovereign AI with Paradom: A Practical Guide

Step 1: Choose your base model
  → Select an open-source model that performs well on your language/domain
  → Recommendation: LLaMA 3 8B or Mistral 7B for general-purpose

Step 2: Design your architecture
  → Optimize for YOUR hardware (consumer GPUs, edge devices)
  → Define your architecture YAML

Step 3: Convert with AWFE
  → Paradom convert --source <chosen-model> --target-arch custom \
      --target-config ./my_country_arch.yaml

Step 4: Fine-tune on local data (optional but recommended)
  → Use standard LoRA fine-tuning with local language data
  → Paradom gives you the starting point; fine-tuning makes it yours

Step 5: Deploy on sovereign infrastructure
  → Your model, your hardware, your control
```

### Target Sovereign Partners (Year 1)

- Ethiopian AI research institutions (primary target 🇪🇹)
- African Union AI initiative
- Academic research groups in the Global South
- Open-source AI organizations

---

## Community Building

### Open Source Strategy

- **Apache 2.0 license** — fully free, commercial use permitted
- **Architecture Zoo:** Community-contributed architecture definitions
- **Conversion Marketplace:** Share and discover pre-validated conversion configs
- **Leaderboard:** Public quality scores for common source→target pairs

### Academic Publication

Target venues:
1. NeurIPS (primary) — systems track
2. ICLR — efficiency track
3. ACL — if we demonstrate multilingual/sovereignty angle

**Paper title (draft):** "AWFE: Adaptive Weight Fusion for Cross-Architecture Neural Network Transfer"

---

## Long-Term Vision (Week 6+)

| Milestone | Target Date |
|---|---|
| Support for 10+ source architectures | Week 8 |
| Automated architecture design ("what target arch fits my hardware?") | Week 10 |
| Paradom-powered national AI launch (first sovereign deployment) | Week 12 |
| 5000+ GitHub stars | Week 12 |
| Academic paper accepted | Week 10 |
| Commercial hosted service (optional) | Week 16 |
