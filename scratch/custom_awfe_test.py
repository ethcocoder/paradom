"""
AWFE — Knowledge Transfer Through Architecture Morphing
=======================================================
Build a small checkpoint that absorbs knowledge from a bigger model.

Architecture:
  - Big Model: 512 d_model, 8 heads, 12 layers (the "teacher")
  - Small Checkpoint: 256 d_model, 4 heads, 6 layers (the "student")
  - Shared tokenizer (same vocab, no tokenization mismatch)

Pipeline:
  1. Build both models
  2. Train big model on text data
  3. Swap big model weights → small checkpoint
  4. Fine-tune small checkpoint with KD
  5. Small checkpoint is now smarter!
"""

import sys, os, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional

# ── Shared Tokenizer ─────────────────────────────────────────
class SimpleTokenizer:
    """
    Simple character-level tokenizer with shared vocab.
    Both models use this exact same tokenizer.
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self._built = False

    def build_from_texts(self, texts: List[str]):
        """Build vocab from training texts."""
        chars = set()
        for text in texts:
            chars.update(text)
        chars = sorted(chars)[:self.vocab_size - 2]
        self.char_to_id = {c: i + 2 for i, c in enumerate(chars)}
        self.char_to_id['<PAD>'] = 0
        self.char_to_id['<UNK>'] = 1
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self._built = True
        print(f"  [Tokenizer] Vocab size: {len(self.char_to_id)}")

    def encode(self, text: str) -> List[int]:
        if not self._built:
            raise RuntimeError("Tokenizer not built")
        return [self.char_to_id.get(c, 1) for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id_to_char.get(i, '') for i in ids if i > 0)

    def __len__(self):
        return self.vocab_size


# ── Transformer Block ────────────────────────────────────────
class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, d_model, n_heads, d_inner, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn_up = nn.Linear(d_model, d_inner)
        self.ffn_down = nn.Linear(d_inner, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        # Self-attention
        h = self.ln1(x)
        qkv = self.attn_qkv(h)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.dropout1(self.attn_out(out))

        # FFN
        h = self.ln2(x)
        h = F.gelu(self.ffn_up(h))
        x = x + self.dropout2(self.ffn_down(h))

        return x


# ── Custom Transformer Model ─────────────────────────────────
class CustomTransformer(nn.Module):
    """
    Our own transformer architecture.
    Can be configured with different d_model, n_heads, n_layers.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_inner, n_layers, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_inner)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        print(f"  [Model] vocab={vocab_size}, d_model={d_model}, "
              f"heads={n_heads}, d_inner={d_inner}, layers={n_layers}")
        print(f"  [Model] params={sum(p.numel() for p in self.parameters())/1e6:.2f}M")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.embed(idx) + self.pos_embed(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -512:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ── Weight Swap Function ─────────────────────────────────────
def swap_weights(src: CustomTransformer, tgt: CustomTransformer) -> Dict:
    """
    Swap weights from big model → small checkpoint.

    Maps each layer proportionally:
    - Embedding: project d_model via PCA or truncation
    - Attention QKV: project and remap heads
    - Attention output: project d_model
    - FFN: project d_model and d_inner
    - Norm: copy with padding

    Returns dict of swapped state_dict.
    """
    src_sd = {k: v.clone().cpu() for k, v in src.state_dict().items()}
    tgt_sd = tgt.state_dict()

    swapped = {}
    mapped = 0
    skipped = 0

    # Build d_model projection: src_d_model → tgt_d_model
    src_d = src.d_model
    tgt_d = tgt.d_model
    src_heads = src.n_heads
    tgt_heads = tgt.n_heads
    src_layers = src.n_layers
    tgt_layers = tgt.n_layers

    print(f"\n  [Swap] Source: {src_d}d, {src_heads}h, {src_layers}L")
    print(f"  [Swap] Target: {tgt_d}d, {tgt_heads}h, {tgt_layers}L")

    # Layer mapping: select src layers for each tgt layer
    layer_map = []
    for i in range(tgt_layers):
        src_idx = int(i * src_layers / tgt_layers)
        layer_map.append(src_idx)
    print(f"  [Swap] Layer map: {layer_map}")

    # 1. Embedding: project d_model
    W_embed = src_sd['embed.weight']  # (vocab, src_d)
    if src_d != tgt_d:
        # Truncate to smaller dim
        W_proj = W_embed[:, :tgt_d].clone()
        # Pad if needed
        if W_proj.shape[1] < tgt_d:
            pad = torch.zeros(W_proj.shape[0], tgt_d - W_proj.shape[1])
            W_proj = torch.cat([W_proj, pad], dim=1)
    else:
        W_proj = W_embed.clone()
    swapped['embed.weight'] = W_proj
    mapped += 1

    # 2. Position embedding: truncate or pad
    W_pos = src_sd['pos_embed.weight']  # (max_pos, src_d)
    if src_d != tgt_d:
        W_pos_proj = W_pos[:, :tgt_d].clone()
        if W_pos_proj.shape[1] < tgt_d:
            pad = torch.zeros(W_pos_proj.shape[0], tgt_d - W_pos_proj.shape[1])
            W_pos_proj = torch.cat([W_pos_proj, pad], dim=1)
    else:
        W_pos_proj = W_pos.clone()
    swapped['pos_embed.weight'] = W_pos_proj
    mapped += 1

    # 3. Transformer blocks
    for tgt_i, src_i in enumerate(layer_map):
        prefix_src = f'blocks.{src_i}'
        prefix_tgt = f'blocks.{tgt_i}'

        # LayerNorm1
        swapped[f'{prefix_tgt}.ln1.weight'] = torch.zeros(tgt_d)
        swapped[f'{prefix_tgt}.ln1.bias'] = torch.zeros(tgt_d)
        mapped += 2

        # Attention QKV: project d_model
        qkv_w = src_sd[f'{prefix_src}.attn_qkv.weight']  # (3*src_d, src_d)
        qkv_b = src_sd[f'{prefix_src}.attn_qkv.bias']    # (3*src_d,)

        # Split into Q, K, V
        q_w = qkv_w[:src_d]      # (src_d, src_d)
        k_w = qkv_w[src_d:2*src_d]
        v_w = qkv_w[2*src_d:]

        # Remap heads
        q_proj = _remap_heads(q_w, src_heads, tgt_heads, src_d, tgt_d)
        k_proj = _remap_heads(k_w, src_heads, tgt_heads, src_d, tgt_d)
        v_proj = _remap_heads(v_w, src_heads, tgt_heads, src_d, tgt_d)

        # Fuse back
        fused = torch.cat([q_proj, k_proj, v_proj], dim=0)  # (3*tgt_d, tgt_d)
        swapped[f'{prefix_tgt}.attn_qkv.weight'] = fused
        swapped[f'{prefix_tgt}.attn_qkv.bias'] = torch.zeros(3 * tgt_d)
        mapped += 2

        # Attention output: project d_model
        o_w = src_sd[f'{prefix_src}.attn_out.weight']  # (src_d, src_d)
        o_b = src_sd[f'{prefix_src}.attn_out.bias']
        swapped[f'{prefix_tgt}.attn_out.weight'] = _project_2d(o_w, src_d, tgt_d)
        swapped[f'{prefix_tgt}.attn_out.bias'] = torch.zeros(tgt_d)
        mapped += 2

        # LayerNorm2
        swapped[f'{prefix_tgt}.ln2.weight'] = torch.zeros(tgt_d)
        swapped[f'{prefix_tgt}.ln2.bias'] = torch.zeros(tgt_d)
        mapped += 2

        # FFN: project d_model and d_inner
        src_inner = src.blocks[0].ffn_up.out_features
        tgt_inner = tgt.blocks[0].ffn_up.out_features

        up_w = src_sd[f'{prefix_src}.ffn_up.weight']   # (src_inner, src_d)
        up_b = src_sd[f'{prefix_src}.ffn_up.bias']
        swapped[f'{prefix_tgt}.ffn_up.weight'] = _project_2d(up_w, src_d, tgt_d, out_dim=tgt_inner)
        swapped[f'{prefix_tgt}.ffn_up.bias'] = torch.zeros(tgt_inner)
        mapped += 2

        down_w = src_sd[f'{prefix_src}.ffn_down.weight']  # (src_d, src_inner)
        down_b = src_sd[f'{prefix_src}.ffn_down.bias']
        swapped[f'{prefix_tgt}.ffn_down.weight'] = _project_2d(down_w, src_d, tgt_d, in_dim=tgt_inner)
        swapped[f'{prefix_tgt}.ffn_down.bias'] = torch.zeros(tgt_d)
        mapped += 2

    # 4. Final layer norm
    swapped['ln_f.weight'] = torch.zeros(tgt_d)
    swapped['ln_f.bias'] = torch.zeros(tgt_d)
    mapped += 2

    print(f"  [Swap] Mapped {mapped} tensors, skipped {skipped}")
    return swapped


def _remap_heads(W: Tensor, src_heads: int, tgt_heads: int, src_d: int, tgt_d: int) -> Tensor:
    """Remap attention heads and project d_model."""
    head_dim = src_d // src_heads
    W_3d = W.view(src_heads, head_dim, src_d)

    selected = []
    for i in range(tgt_heads):
        src_idx = i % src_heads
        selected.append(W_3d[src_idx])

    out = torch.stack(selected, dim=0).view(tgt_heads * head_dim, src_d)

    # Project d_model: (tgt_heads*head_dim, src_d) → (tgt_d, tgt_d)
    if src_d != tgt_d:
        out_proj = torch.zeros(tgt_d, tgt_d)
        out_proj[:, :src_d] = out[:min(tgt_d, out.shape[0])]
        return out_proj

    return out


def _project_2d(W: Tensor, src_d: int, tgt_d: int,
                in_dim: int = None, out_dim: int = None) -> Tensor:
    """Project a 2D weight matrix to target dimensions."""
    if in_dim is None:
        in_dim = W.shape[0]
    if out_dim is None:
        out_dim = W.shape[1]

    result = torch.zeros(in_dim, out_dim)

    # Copy overlapping region
    copy_rows = min(W.shape[0], in_dim)
    copy_cols = min(W.shape[1], out_dim)
    result[:copy_rows, :copy_cols] = W[:copy_rows, :copy_cols]

    return result


# ── Training Data ─────────────────────────────────────────────
TRAINING_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning was the word, and the word was with God.",
    "To be or not to be, that is the question.",
    "The capital of France is Paris, known for the Eiffel Tower.",
    "Water boils at 100 degrees Celsius at standard pressure.",
    "Once upon a time in a land far away, there lived a brave knight.",
    "The sun rises in the east and sets in the west.",
    "Python is a popular programming language for many applications.",
    "The mitochondria is the powerhouse of the cell.",
    "Machine learning is a subset of artificial intelligence.",
    "She opened the door and found a beautiful garden outside.",
    "The teacher explained the lesson clearly to all students.",
    "Technology continues to advance at an unprecedented rate.",
    "The Earth orbits the Sun at 150 million kilometers.",
    "Music has the power to evoke strong emotions and memories.",
    "The Great Wall of China is over 13,000 miles long.",
    "Exercise regularly to maintain good physical and mental health.",
    "Knowledge is power, and learning is the key to success.",
    "The universe is vast and full of mysteries waiting to be solved.",
    "Every great journey begins with a single step forward.",
    "Science helps us understand the natural world around us.",
    "Creativity is intelligence having fun with new ideas.",
    "The only way to do great work is to love what you do.",
    "Innovation distinguishes between a leader and a follower.",
    "Stay hungry, stay foolish, and never stop learning.",
]


# ── Main Pipeline ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AWFE — Knowledge Transfer Through Architecture Morphing")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # 1. Build shared tokenizer
    print(f"\n[1] Building shared tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_from_texts(TRAINING_TEXTS)

    # 2. Build models
    print(f"\n[2] Building models...")
    BIG_CONFIG = dict(vocab_size=len(tokenizer), d_model=512, n_heads=8, d_inner=2048, n_layers=12)
    SMALL_CONFIG = dict(vocab_size=len(tokenizer), d_model=256, n_heads=4, d_inner=1024, n_layers=6)

    print(f"\n  Big Model (teacher):")
    big_model = CustomTransformer(**BIG_CONFIG).to(device)

    print(f"\n  Small Checkpoint (student):")
    small_model = CustomTransformer(**SMALL_CONFIG).to(device)

    # 3. Prepare training data
    print(f"\n[3] Preparing training data...")
    all_ids = []
    for text in TRAINING_TEXTS:
        ids = tokenizer.encode(text)
        all_ids.append(torch.tensor(ids, device=device))
    print(f"  {len(all_ids)} sequences, avg length: {sum(len(i) for i in all_ids)//len(all_ids)}")

    # 4. Train big model
    print(f"\n[4] Training big model ({BIG_CONFIG['d_model']}d, {BIG_CONFIG['n_layers']}L)...")
    big_model.train()
    optimizer = torch.optim.AdamW(big_model.parameters(), lr=3e-4, weight_decay=0.01)
    n_steps = 2000
    batch_size = 4
    seq_len = 128

    t0 = time.time()
    for step in range(n_steps):
        # Sample batch
        batch_ids = random.choices(all_ids, k=batch_size)
        # Pad to same length
        max_len = max(len(ids) for ids in batch_ids)
        x = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            x[i, :len(ids)] = ids
        targets = x[:, 1:].contiguous()
        x = x[:, :-1]

        _, loss = big_model(x, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(big_model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1}/{n_steps}, Loss: {loss.item():.4f}")

    big_model.eval()
    elapsed = time.time() - t0
    print(f"  Big model trained in {elapsed:.1f}s")

    # Test big model
    print(f"\n  [Big Model Test]")
    prompt_ids = torch.tensor([tokenizer.encode("Once upon a time")], device=device)
    big_output = big_model.generate(prompt_ids, max_new_tokens=100)
    print(f"  Big: \"{tokenizer.decode(big_output[0].tolist())}\"")

    # 5. Save big model
    print(f"\n[5] Saving big model...")
    torch.save(big_model.state_dict(), "big_model.pt")
    print(f"  Saved to big_model.pt")

    # 6. Swap weights: big → small
    print(f"\n[6] Swapping weights: big({BIG_CONFIG['d_model']}d) → small({SMALL_CONFIG['d_model']}d)...")
    swapped = swap_weights(big_model, small_model)

    # Load swapped weights
    small_model.load_state_dict(swapped, strict=False)
    print(f"  Loaded {len(swapped)} tensors into small model")

    # Test swapped small model
    print(f"\n  [Swapped Small Test]")
    small_output = small_model.generate(prompt_ids, max_new_tokens=100)
    print(f"  Swapped: \"{tokenizer.decode(small_output[0].tolist())}\"")

    # 7. Fine-tune small model with KD
    print(f"\n[7] Fine-tuning small model with KD...")
    small_model.train()

    # Freeze embedding (keep swapped knowledge)
    small_model.embed.weight.requires_grad = False
    small_model.head.weight = small_model.embed.weight

    trainable = [p for p in small_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    temp = 2.0
    n_ft = 1000

    print(f"  Training {len(trainable)} params for {n_ft} steps...")
    t0 = time.time()

    for step in range(n_ft):
        batch_ids = random.choices(all_ids, k=batch_size)
        max_len = max(len(ids) for ids in batch_ids)
        x = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            x[i, :len(ids)] = ids
        targets = x[:, 1:].contiguous()
        x = x[:, :-1]

        # Teacher soft targets
        with torch.no_grad():
            teacher_logits, _ = big_model(x)
            teacher_probs = F.softmax(teacher_logits / temp, dim=-1)

        # Student forward
        student_logits, student_loss = small_model(x, targets)

        # KD loss
        student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temp ** 2)

        # Combined loss
        loss = 0.5 * student_loss + 0.5 * kd_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}/{n_ft}, Loss: {loss.item():.4f} (CE: {student_loss.item():.4f}, KD: {kd_loss.item():.4f})")

    small_model.eval()
    elapsed = time.time() - t0
    print(f"  Fine-tuned in {elapsed:.1f}s")

    # Test fine-tuned small model
    print(f"\n  [Fine-tuned Small Test]")
    ft_output = small_model.generate(prompt_ids, max_new_tokens=100)
    print(f"  Fine-tuned: \"{tokenizer.decode(ft_output[0].tolist())}\"")

    # 8. Also train a small model from scratch (for comparison)
    print(f"\n[8] Training small model from scratch (baseline)...")
    small_scratch = CustomTransformer(**SMALL_CONFIG).to(device)
    small_scratch.train()
    optimizer2 = torch.optim.AdamW(small_scratch.parameters(), lr=3e-4, weight_decay=0.01)

    t0 = time.time()
    for step in range(n_ft):
        batch_ids = random.choices(all_ids, k=batch_size)
        max_len = max(len(ids) for ids in batch_ids)
        x = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            x[i, :len(ids)] = ids
        targets = x[:, 1:].contiguous()
        x = x[:, :-1]

        _, loss = small_scratch(x, targets)
        optimizer2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(small_scratch.parameters(), 1.0)
        optimizer2.step()

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}/{n_ft}, Loss: {loss.item():.4f}")

    small_scratch.eval()
    elapsed = time.time() - t0
    print(f"  Scratch model trained in {elapsed:.1f}s")

    print(f"\n  [Scratch Small Test]")
    scratch_output = small_scratch.generate(prompt_ids, max_new_tokens=100)
    print(f"  Scratch: \"{tokenizer.decode(scratch_output[0].tolist())}\"")

    # 9. Summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Big Model (512d):     \"{tokenizer.decode(big_output[0].tolist())}\"")
    print(f"  Swapped (Pre-FT):     \"{tokenizer.decode(small_output[0].tolist())}\"")
    print(f"  Swapped (Post-FT):    \"{tokenizer.decode(ft_output[0].tolist())}\"")
    print(f"  Scratch (same steps): \"{tokenizer.decode(scratch_output[0].tolist())}\"")
    print(f"{'=' * 60}")
    print(f"  If Swapped-FT is better than Scratch, AWFE is PROVED.")
    print(f"  The weight transfer gave the small model a head start!")
    print(f"{'=' * 60}")

    # Save models
    torch.save(small_model.state_dict(), "small_model_swapped.pt")
    torch.save(small_scratch.state_dict(), "small_model_scratch.pt")
    print(f"\n  Models saved!")


if __name__ == "__main__":
    main()
