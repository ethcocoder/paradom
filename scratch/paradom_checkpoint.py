"""
Paradom Checkpoint — Custom Architecture for AWFE Proof
========================================================
Build a small custom transformer with DIFFERENT architecture from SmolLM,
but SAME tokenizer. Then swap SmolLM's knowledge into it.

Architecture differences from SmolLM:
- d_model: 256 (vs 576)
- n_heads: 4 (vs 9), head_dim: 64
- d_inner: 512 (vs 1536)
- n_layers: 6 (vs 30)
- FFN: GELU (not SwiGLU)
- Attention: MHA (not GQA)
- Position: Sinusoidal (not RoPE)
- Vocab: Same as SmolLM (49152) via shared tokenizer

The swap proves: different architecture + same tokenizer = AWFE works.
"""

import sys, os, time, math, re
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuration ────────────────────────────────────────────
SMOLM_ID = "HuggingFaceTB/SmolLM-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Our custom model config (DIFFERENT from SmolLM)
OUR_CONFIG = {
    "vocab_size": 49152,  # Same as SmolLM
    "d_model": 256,       # vs SmolLM's 576
    "n_heads": 4,         # vs SmolLM's 9
    "head_dim": 64,       # Same head dim
    "d_inner": 512,       # vs SmolLM's 1536
    "n_layers": 6,        # vs SmolLM's 30
    "max_seq_len": 512,
    "dropout": 0.1,
}

# Training config
TRAIN_CONFIG = {
    "batch_size": 8,
    "lr": 3e-4,
    "n_steps": 2000,
    "warmup_steps": 100,
    "kd_steps": 1000,      # KD fine-tuning after swap
    "kd_alpha": 0.7,       # KD weight
    "kd_temp": 2.0,        # KD temperature
}


# ═══════════════════════════════════════════════════════════════
# OUR CUSTOM MODEL
# ═══════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position encoding (different from SmolLM's RoPE)."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (NOT GQA like SmolLM — different architecture)."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            att = att.masked_fill(mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class GELUFFN(nn.Module):
    """GELU FFN (NOT SwiGLU like SmolLM — different architecture)."""
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GELU FFN."""
    def __init__(self, d_model, n_heads, d_inner, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = GELUFFN(d_model, d_inner, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class ParadomModel(nn.Module):
    """
    Our custom transformer — DIFFERENT architecture from SmolLM.
    Same tokenizer, different structure.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_inner = config["d_inner"]
        self.n_layers = config["n_layers"]

        # Embedding (same vocab as SmolLM)
        self.tok_emb = nn.Embedding(config["vocab_size"], self.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, config["max_seq_len"])
        self.drop = nn.Dropout(config["dropout"])

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_inner, config["dropout"])
            for _ in range(self.n_layers)
        ])

        # Final norm + output
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, config["vocab_size"], bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

        x = self.tok_emb(idx) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# ═══════════════════════════════════════════════════════════════
# WEIGHT SWAP: SmolLM → Our Model
# ═══════════════════════════════════════════════════════════════

def extract_smolm_weights(model):
    """Extract SmolLM state dict with role labels."""
    sd = {}
    for k, v in model.model.state_dict().items():
        sd[k] = v.float().cpu()
    return sd


def swap_smolm_to_ours(smolm_sd, our_config):
    """
    Map SmolLM weights → our model's architecture.

    Key mappings:
    - embed_tokens → tok_emb (project d_model 576→256 via PCA)
    - q/k/v_proj → q/k/v_proj (project d_model + head remap 9→4)
    - o_proj → o_proj (project d_model 576→256)
    - gate_proj + up_proj → fc1 (SwiGLU→GELU, project d_model)
    - down_proj → fc2 (project d_model)
    - input_layernorm → ln1
    - post_attention_layernorm → ln2
    - norm → ln_f
    """
    our_sd = {}
    d_src = 576
    d_tgt = our_config["d_model"]
    h_src = 9
    h_tgt = our_config["n_heads"]
    head_dim = 64
    d_inner_src = 1536
    d_inner_tgt = our_config["d_inner"]

    print(f"  [Swap] Mapping SmolLM ({d_src}d, {h_src}h, {d_inner_src}d) → "
          f"OurModel ({d_tgt}d, {h_tgt}h, {d_inner_tgt}d)")

    # Build projection matrices
    # For d_model: select top-256 dims by variance from embedding
    emb = smolm_sd["model.embed_tokens.weight"]  # (49152, 576)
    var = emb.var(dim=0)  # (576,)
    _, top_idx = var.topk(d_tgt)
    top_idx = top_idx.sort()[0]
    P = torch.zeros(d_src, d_tgt)
    for i, idx in enumerate(top_idx):
        P[idx, i] = 1.0

    # 1. Token embedding
    our_sd["tok_emb.weight"] = emb[:, top_idx]  # (49152, 256)

    # 2. Transformer layers (select first 6 of 30)
    src_layers = list(range(6))

    for tgt_i, src_i in enumerate(src_layers):
        # Pre-attention norm
        ln1_w = smolm_sd[f"model.layers.{src_i}.input_layernorm.weight"]
        our_sd[f"blocks.{tgt_i}.ln1.weight"] = ln1_w[:d_tgt].clone()
        our_sd[f"blocks.{tgt_i}.ln1.bias"] = torch.zeros(d_tgt)

        # Q projection
        q_w = smolm_sd[f"model.layers.{src_i}.self_attn.q_proj.weight"]  # (576, 576)
        q_proj = (q_w @ P)[:, top_idx]  # (576, 256) → select top-256 rows
        # Remap heads: 9 → 4 (keep first 4 heads)
        q_3d = q_proj.reshape(h_src, head_dim, d_tgt)  # (9, 64, 256)
        q_3d = q_3d[:h_tgt]  # (4, 64, 256)
        our_sd[f"blocks.{tgt_i}.attn.q_proj.weight"] = q_3d.reshape(h_tgt * head_dim, d_tgt)

        # K projection
        k_w = smolm_sd[f"model.layers.{src_i}.self_attn.k_proj.weight"]  # (192, 576)
        k_proj = (k_w @ P)[:, top_idx]  # (192, 256)
        # Remap: 3 KV heads → 4 heads (cycle: 0,1,2,0)
        k_3d = k_proj.reshape(3, head_dim, d_tgt)  # (3, 64, 256)
        k_remapped = torch.stack([k_3d[i % 3] for i in range(h_tgt)])  # (4, 64, 256)
        our_sd[f"blocks.{tgt_i}.attn.k_proj.weight"] = k_remapped.reshape(h_tgt * head_dim, d_tgt)

        # V projection
        v_w = smolm_sd[f"model.layers.{src_i}.self_attn.v_proj.weight"]
        v_proj = (v_w @ P)[:, top_idx]
        v_3d = v_proj.reshape(3, head_dim, d_tgt)
        v_remapped = torch.stack([v_3d[i % 3] for i in range(h_tgt)])
        our_sd[f"blocks.{tgt_i}.attn.v_proj.weight"] = v_remapped.reshape(h_tgt * head_dim, d_tgt)

        # O projection
        o_w = smolm_sd[f"model.layers.{src_i}.self_attn.o_proj.weight"]  # (576, 576)
        o_proj = P.T @ o_w @ P  # (256, 256)
        our_sd[f"blocks.{tgt_i}.attn.o_proj.weight"] = o_proj

        # Post-attention norm
        ln2_w = smolm_sd[f"model.layers.{src_i}.post_attention_layernorm.weight"]
        our_sd[f"blocks.{tgt_i}.ln2.weight"] = ln2_w[:d_tgt].clone()
        our_sd[f"blocks.{tgt_i}.ln2.bias"] = torch.zeros(d_tgt)

        # FFN: SwiGLU (gate+up) → GELU (fc1)
        gate_w = smolm_sd[f"model.layers.{src_i}.mlp.gate_proj.weight"]  # (1536, 576)
        up_w = smolm_sd[f"model.layers.{src_i}.mlp.up_proj.weight"]      # (1536, 576)
        # Take first d_inner_tgt rows of gate (simple truncation)
        fc1 = gate_w[:d_inner_tgt] @ P  # (512, 256)
        our_sd[f"blocks.{tgt_i}.ffn.fc1.weight"] = fc1[:, top_idx]

        # FFN down: down_proj → fc2
        down_w = smolm_sd[f"model.layers.{src_i}.mlp.down_proj.weight"]  # (576, 1536)
        fc2 = P.T @ down_w[:, :d_inner_tgt]  # (256, 512)
        our_sd[f"blocks.{tgt_i}.ffn.fc2.weight"] = fc2

    # 3. Final norm
    norm_w = smolm_sd["model.norm.weight"]
    our_sd["ln_f.weight"] = norm_w[:d_tgt].clone()
    our_sd["ln_f.bias"] = torch.zeros(d_tgt)

    # 4. Skip tok_emb.weight (tied to head, already set)

    return our_sd


# ═══════════════════════════════════════════════════════════════
# FINE-TUNING
# ═══════════════════════════════════════════════════════════════

CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning was the word, and the word was with God.",
    "To be or not to be, that is the question.",
    "The capital of France is Paris, known for the Eiffel Tower.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Once upon a time in a land far away, there lived a brave knight.",
    "The sun rises in the east and sets in the west.",
    "Python is a popular programming language for many applications.",
    "The mitochondria is the powerhouse of the cell.",
    "Machine learning is a subset of artificial intelligence.",
    "She opened the door and found a beautiful garden outside.",
    "The teacher explained the lesson clearly to all students.",
    "Technology continues to advance at an unprecedented rate.",
    "The Earth orbits the Sun at an average distance of 150 million kilometers.",
    "Music has the power to evoke strong emotions and memories.",
    "The Great Wall of China is over 13,000 miles long.",
    "Exercise regularly to maintain good physical and mental health.",
    "The weather forecast predicts rain for tomorrow afternoon.",
    "He read a book about the history of ancient civilizations.",
    "The dog barked loudly at the stranger approaching the house.",
]


def train_large_model(model, tokenizer, texts, config):
    """Train the large model (teacher) on calibration data."""
    model.train()
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # Tokenize
    all_ids = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", max_length=256,
                       truncation=True).input_ids.squeeze(0)
        all_ids.append(ids)

    print(f"  [Train Large] {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"  [Train Large] Training for {config['n_steps']} steps...")

    t0 = time.time()
    losses = []
    for step in range(config["n_steps"]):
        idx = step % len(all_ids)
        input_ids = all_ids[idx].unsqueeze(0).to(DEVICE)

        logits, loss = model(input_ids, targets=input_ids)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            avg = sum(losses[-500:]) / len(losses[-500:])
            print(f"  [Train Large] Step {step+1}/{config['n_steps']}, Avg Loss: {avg:.4f}")

    elapsed = time.time() - t0
    print(f"  [Train Large] Done in {elapsed:.1f}s")
    model.eval()
    return model


def fine_tune_with_kd(student, teacher, tokenizer, texts, config):
    """Fine-tune student with knowledge distillation from teacher."""
    student.train()
    student.to(DEVICE)
    teacher.eval()
    teacher.to(DEVICE)

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    temp = config["kd_temp"]
    alpha = config["kd_alpha"]

    # Tokenize
    all_ids = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", max_length=256,
                       truncation=True).input_ids.squeeze(0)
        all_ids.append(ids)

    print(f"  [KD] Training for {config['kd_steps']} steps (alpha={alpha}, temp={temp})...")
    t0 = time.time()

    for step in range(config["kd_steps"]):
        idx = step % len(all_ids)
        input_ids = all_ids[idx].unsqueeze(0).to(DEVICE)

        # Teacher logits
        with torch.no_grad():
            teacher_logits, _ = teacher(input_ids)

        # Student logits
        student_logits, ce_loss = student(input_ids, targets=input_ids)

        # KD loss
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / temp, dim=-1),
            F.softmax(teacher_logits / temp, dim=-1),
            reduction='batchmean'
        ) * (temp ** 2)

        loss = alpha * kd_loss + (1 - alpha) * ce_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"  [KD] Step {step+1}/{config['kd_steps']}, Loss: {loss.item():.4f} "
                  f"(CE: {ce_loss.item():.4f}, KD: {kd_loss.item():.4f})")

    elapsed = time.time() - t0
    print(f"  [KD] Done in {elapsed:.1f}s")
    student.eval()
    return student


def generate(model, tokenizer, prompt, max_new=100):
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(ids)
            next_logits = logits[:, -1, :] / 0.8
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PARADOM CHECKPOINT — Custom Architecture AWFE Proof")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    # 1. Load SmolLM tokenizer + model
    print(f"\n[1] Loading {SMOLM_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(SMOLM_ID)
    smolm_model = AutoModelForCausalLM.from_pretrained(
        SMOLM_ID, dtype=torch.float32
    ).to(DEVICE).eval()
    print(f"  Loaded | {sum(p.numel() for p in smolm_model.parameters())/1e6:.1f}M params")

    # 2. Build our custom model
    print(f"\n[2] Building our custom model ({OUR_CONFIG['d_model']}d, {OUR_CONFIG['n_heads']}h, "
          f"{OUR_CONFIG['d_inner']}d_inner, {OUR_CONFIG['n_layers']}L)...")
    our_model = ParadomModel(OUR_CONFIG).to(DEVICE)
    print(f"  Built | {sum(p.numel() for p in our_model.parameters())/1e6:.2f}M params")

    # 3. Extract SmolLM weights
    print(f"\n[3] Extracting SmolLM weights...")
    smolm_sd = extract_smolm_weights(smolm_model)
    del smolm_model

    # 4. Swap weights
    print(f"\n[4] Swapping SmolLM → Our Model...")
    our_sd = swap_smolm_to_ours(smolm_sd, OUR_CONFIG)

    # Load into our model
    our_model.load_state_dict(our_sd, strict=False)
    print(f"  Loaded {len(our_sd)} tensors")

    # 5. Test swapped model
    print(f"\n[5] Testing swapped model...")
    pre_ft = generate(our_model, tokenizer, "Once upon a time in a land far away,")
    print(f"  [Pre-FT] \"{pre_ft}\"")

    # 6. Fine-tune with KD
    print(f"\n[6] Fine-tuning with KD from SmolLM...")
    # Reload teacher
    teacher_model = AutoModelForCausalLM.from_pretrained(
        SMOLM_ID, dtype=torch.float32
    ).to(DEVICE).eval()

    our_model = fine_tune_with_kd(our_model, teacher_model, tokenizer,
                                   CALIBRATION_TEXTS, TRAIN_CONFIG)

    # 7. Test after fine-tuning
    print(f"\n[7] Testing after fine-tuning...")
    post_ft = generate(our_model, tokenizer, "Once upon a time in a land far away,")
    print(f"  [Post-FT] \"{post_ft}\"")

    # 8. Test more prompts
    test_prompts = [
        "The meaning of life is",
        "In science, we learn that",
        "Hello, my name is",
        "The best way to learn is",
    ]
    for prompt in test_prompts:
        out = generate(our_model, tokenizer, prompt)
        print(f"\n  Q: {prompt}")
        print(f"  A: {out}")

    # 9. Summary
    print(f"\n{'=' * 60}")
    print(f"  AWFE PROOF: Custom Architecture")
    print(f"{'=' * 60}")
    print(f"  Architecture: DIFFERENT from SmolLM")
    print(f"  Tokenizer: SAME as SmolLM")
    print(f"  Knowledge: SWAPPED from SmolLM")
    print(f"  Fine-tuned: YES (KD from SmolLM)")
    print(f"{'=' * 60}")

    # Save
    torch.save(our_model.state_dict(), "paradom_checkpoint.pt")
    print(f"  Model saved to paradom_checkpoint.pt")


if __name__ == "__main__":
    main()
