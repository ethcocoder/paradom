"""
AWFE Phase 2 — Fine-Tuning Recovery Test v2
=============================================
Key improvements:
1. Knowledge distillation: source model guides swapped model
2. Cosine LR schedule with warmup
3. 500 steps instead of 200
4. Larger calibration set (20 texts)
"""

import sys, os, time, re, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from paradom.core.types import WeightProduct
from paradom.core.enums import FunctionalRole
from paradom.core.matcher import FunctionalRoleMatcher
from paradom.mappings.transformer_to_transformer import TransformerToTransformerMapper

# ── Configuration ────────────────────────────────────────────
MODEL_ID     = "HuggingFaceTB/SmolLM-135M"
PROMPT       = "Once upon a time in a land far away,"
MAX_NEW_TOK  = 50
DEVICE       = "cpu"

SOURCE_CONFIG = {
    "d_model": 576, "d_inner": 1536,
    "num_heads": 9, "num_key_value_heads": 3,
    "head_dim": 64, "num_hidden_layers": 30,
    "vocab_size": 49152,
}

TARGET_B = {
    "d_model": 512, "d_inner": 1408,
    "num_heads": 8, "num_key_value_heads": 2,
    "head_dim": 64, "num_hidden_layers": 30,
    "vocab_size": 49152,
}

CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning was the word, and the word was with God.",
    "To be or not to be, that is the question.",
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The quick brown fox jumps over the lazy dog. This is a test.",
    "In a hole in the ground there lived a hobbit.",
    "The sun rises in the east and sets in the west.",
    "Python is a popular programming language used for many applications.",
    "The mitochondria is the powerhouse of the cell.",
    "Machine learning is a subset of artificial intelligence.",
    "The Earth orbits the Sun at an average distance of 150 million kilometers.",
    "Music has the power to evoke strong emotions and memories.",
    "The Great Wall of China is over 13,000 miles long.",
    "Exercise regularly to maintain good physical and mental health.",
    "Once upon a time in a land far away, there lived a brave knight.",
    "The weather forecast predicts rain for tomorrow afternoon.",
    "She opened the door and found a beautiful garden outside.",
    "The teacher explained the lesson clearly to all the students.",
    "Technology continues to advance at an unprecedented rate.",
]

# ── Helpers ──────────────────────────────────────────────────
matcher = FunctionalRoleMatcher()

def state_dict_to_weight_products(sd, architecture="llama"):
    products = []
    for key, tensor in sd.items():
        role = matcher.assign_role(key, architecture)
        layer_idx = -1
        m = re.search(r"layers\.(\d+)\.", key)
        if m: layer_idx = int(m.group(1))
        products.append(WeightProduct(
            name=key, tensor=tensor.float(),
            shape=tuple(tensor.shape), functional_role=role,
            paradigm="transformer", architecture=architecture,
            layer_index=layer_idx, dtype=tensor.dtype,
        ))
    return products


def generate(model, tokenizer, prompt, max_new=MAX_NEW_TOK):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def build_target_model(target_hf_config):
    cfg = LlamaConfig(
        hidden_size=target_hf_config["d_model"],
        intermediate_size=target_hf_config["d_inner"],
        num_hidden_layers=target_hf_config["num_hidden_layers"],
        num_attention_heads=target_hf_config["num_heads"],
        num_key_value_heads=target_hf_config["num_key_value_heads"],
        vocab_size=target_hf_config["vocab_size"],
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(cfg).to(DEVICE).eval()


def load_swapped_weights(model, swapped_sd):
    model_sd = model.model.state_dict()
    loaded, skipped, mismatched = 0, 0, 0
    for key in model_sd:
        if key in swapped_sd:
            src = swapped_sd[key]
            if src.shape == model_sd[key].shape:
                model_sd[key] = src.to(model_sd[key].dtype)
                loaded += 1
            else:
                mismatched += 1
        else:
            skipped += 1
    model.model.load_state_dict(model_sd, strict=False)
    if hasattr(model, 'lm_head') and hasattr(model.model, 'embed_tokens'):
        model.lm_head.weight = model.model.embed_tokens.weight
    return loaded, skipped, mismatched


def fine_tune_with_distillation(
    student_model, teacher_model, tokenizer, calibration_texts,
    num_steps=500, lr=1e-4, device="cpu",
    alpha=0.5,  # distillation weight (0=pure CE, 1=pure KD)
    temperature=2.0
):
    """
    Fine-tune student model with knowledge distillation from teacher.
    
    Loss = alpha * KD_loss + (1-alpha) * CE_loss
    
    KD_loss: KL divergence between teacher and student soft targets
    CE_loss: Standard cross-entropy on calibration data
    """
    student_model.train()
    teacher_model.eval()
    teacher_model.to(device)
    student_model.to(device)
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Trainable: all transformer layers
    trainable_params = []
    for name, param in student_model.named_parameters():
        if 'layers.' in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"  [KD] {len(trainable_params)} trainable layers, {sum(p.numel() for p in trainable_params)/1e6:.1f}M params")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    # Cosine schedule with warmup
    warmup_steps = min(50, num_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Prepare calibration inputs
    all_input_ids = []
    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=256,
                           truncation=True, padding=False)
        all_input_ids.append(inputs.input_ids.squeeze(0))
    
    print(f"  [KD] Training for {num_steps} steps (alpha={alpha}, temp={temperature})...")
    t0 = time.time()
    
    for step in range(num_steps):
        idx = step % len(all_input_ids)
        input_ids = all_input_ids[idx].unsqueeze(0).to(device)
        
        # Teacher logits (no grad)
        with torch.no_grad():
            teacher_out = teacher_model(input_ids)
            teacher_logits = teacher_out.logits
        
        # Student logits
        student_out = student_model(input_ids, labels=input_ids)
        student_logits = student_out.logits
        ce_loss = student_out.loss
        
        # KD loss: KL divergence between soft targets
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Combined loss
        loss = alpha * kd_loss + (1 - alpha) * ce_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        
        if (step + 1) % 100 == 0:
            print(f"  [KD] Step {step+1}/{num_steps}, Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, KD: {kd_loss.item():.4f})")
    
    elapsed = time.time() - t0
    print(f"  [KD] Done in {elapsed:.1f}s")
    student_model.eval()
    return student_model


# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AWFE Fine-Tuning Recovery Test v2 (KD)")
    print("=" * 60)

    # Load source model (will be teacher)
    print(f"\n[LOAD] {MODEL_ID}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(DEVICE).eval()
    print(f"  Loaded in {time.time()-t0:.1f}s | {sum(p.numel() for p in teacher_model.parameters())/1e6:.1f}M params")

    # Baseline
    baseline = generate(teacher_model, tokenizer, PROMPT)
    print(f"\n[BASELINE] \"{baseline}\"")
    
    # Extract weights
    full_sd = {}
    for k, v in teacher_model.model.state_dict().items():
        full_sd[k] = v.clone()
    products = state_dict_to_weight_products(full_sd)

    # ── Test B: SVD only (no FT) ───────────────────────────
    print(f"\n{'─' * 60}")
    print("  TEST B-SVD: No Fine-Tune (baseline)")
    print(f"{'─' * 60}")
    mapper = TransformerToTransformerMapper(projection_method="svd")
    swapped_sd, eq_map = mapper.convert(products, TARGET_B, swap_fraction=1.0)
    print(f"  Mean CKA: {eq_map.mean_cka:.4f}")
    target_model = build_target_model(TARGET_B)
    load_swapped_weights(target_model, swapped_sd)
    svd_output = generate(target_model, tokenizer, PROMPT)
    print(f"  Output: \"{svd_output}\"")
    del target_model

    # ── Test B: SVD + CE Fine-Tune ─────────────────────────
    print(f"\n{'─' * 60}")
    print("  TEST B-SVD-CE: SVD + CE Fine-Tune (500 steps)")
    print(f"{'─' * 60}")
    mapper = TransformerToTransformerMapper(projection_method="svd")
    swapped_sd, eq_map = mapper.convert(products, TARGET_B, swap_fraction=1.0)
    target_model = build_target_model(TARGET_B)
    load_swapped_weights(target_model, swapped_sd)
    
    pre_ft = generate(target_model, tokenizer, PROMPT)
    print(f"  Pre-FT: \"{pre_ft}\"")
    
    # CE-only fine-tune
    target_model.train()
    trainable = [p for n, p in target_model.named_parameters() if 'layers.' in n]
    for p in trainable: p.requires_grad = True
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    
    all_ids = [tokenizer(t, return_tensors="pt", max_length=256, truncation=True).input_ids.squeeze(0) for t in CALIBRATION_TEXTS]
    
    for step in range(500):
        ids = all_ids[step % len(all_ids)].unsqueeze(0).to(DEVICE)
        out = target_model(ids, labels=ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step+1) % 100 == 0:
            print(f"  Step {step+1}/500, Loss: {loss.item():.4f}")
    
    target_model.eval()
    ce_output = generate(target_model, tokenizer, PROMPT)
    print(f"  Post-CE: \"{ce_output}\"")
    del target_model

    # ── Test B: SVD + KD Fine-Tune ─────────────────────────
    print(f"\n{'─' * 60}")
    print("  TEST B-SVD-KD: SVD + Knowledge Distillation (500 steps)")
    print(f"{'─' * 60}")
    mapper = TransformerToTransformerMapper(projection_method="svd")
    swapped_sd, eq_map = mapper.convert(products, TARGET_B, swap_fraction=1.0)
    target_model = build_target_model(TARGET_B)
    load_swapped_weights(target_model, swapped_sd)
    
    pre_ft = generate(target_model, tokenizer, PROMPT)
    print(f"  Pre-FT: \"{pre_ft}\"")
    
    # KD fine-tune
    target_model = fine_tune_with_distillation(
        target_model, teacher_model, tokenizer, CALIBRATION_TEXTS,
        num_steps=500, lr=1e-4, device=DEVICE,
        alpha=0.7, temperature=2.0
    )
    
    kd_output = generate(target_model, tokenizer, PROMPT)
    print(f"  Post-KD: \"{kd_output}\"")
    del target_model

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:   \"{baseline}\"")
    print(f"  SVD only:   \"{svd_output}\"")
    print(f"  SVD + CE:   \"{ce_output}\"")
    print(f"  SVD + KD:   \"{kd_output}\"")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
