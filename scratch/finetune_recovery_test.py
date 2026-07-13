"""
AWFE Phase 2 — Fine-Tuning Recovery Test
==========================================
After weight swap, fine-tune the model on calibration data
to see if it recovers coherent generation.

Hypothesis: the swapped weights are ~90% correct but in wrong subspace.
A few gradient steps should fix the remaining misalignment.
"""

import sys, os, time, re, copy
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
                print(f"        [MISMATCH] {key}: {src.shape} vs {model_sd[key].shape}")
        else:
            skipped += 1
    model.model.load_state_dict(model_sd, strict=False)
    if hasattr(model, 'lm_head') and hasattr(model.model, 'embed_tokens'):
        model.lm_head.weight = model.model.embed_tokens.weight
    return loaded, skipped, mismatched


def collect_calibration_data(model, tokenizer, calibration_texts, max_length=256, device="cpu"):
    model.eval()
    model.to(device)
    hidden_states = []
    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            hs = output[0]
        elif hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
            hs = output.last_hidden_state
        elif isinstance(output, torch.Tensor):
            hs = output
        else:
            return
        if hs.dim() == 3:
            hs = hs.reshape(-1, hs.shape[-1])
        elif hs.dim() == 2:
            pass
        else:
            return
        hidden_states.append(hs.detach().cpu())
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        last_layer = model.model.layers[-1]
        handle = last_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            for text in calibration_texts:
                inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                                   truncation=True, padding=False)
                input_ids = inputs.input_ids.to(device)
                model(input_ids)
        handle.remove()
    if hidden_states:
        all_hidden = torch.cat(hidden_states, dim=0)
        print(f"  [Calibration] Collected {all_hidden.shape[0]} tokens, d_model={all_hidden.shape[1]}")
        return all_hidden
    return None


def fine_tune_model(model, tokenizer, calibration_texts, num_steps=200, lr=5e-5, device="cpu"):
    """
    Fine-tune the model on calibration data using cross-entropy loss.
    This teaches the model to work in its new dimension space.
    """
    model.train()
    model.to(device)
    
    # Only fine-tune the transformer layers (not embeddings/lm_head if tied)
    trainable_params = []
    for name, param in model.named_parameters():
        if 'layers.' in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"  [FineTune] {len(trainable_params)} trainable layers, {sum(p.numel() for p in trainable_params)/1e6:.1f}M params")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    # Prepare calibration inputs
    all_input_ids = []
    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=256,
                           truncation=True, padding=False)
        all_input_ids.append(inputs.input_ids.squeeze(0))
    
    print(f"  [FineTune] Training for {num_steps} steps...")
    t0 = time.time()
    
    for step in range(num_steps):
        # Pick a random calibration text
        idx = step % len(all_input_ids)
        input_ids = all_input_ids[idx].unsqueeze(0).to(device)
        
        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        
        if (step + 1) % 50 == 0:
            print(f"  [FineTune] Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")
    
    elapsed = time.time() - t0
    print(f"  [FineTune] Done in {elapsed:.1f}s")
    model.eval()
    return model


# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AWFE Fine-Tuning Recovery Test")
    print("=" * 60)

    # Load original model
    print(f"\n[LOAD] {MODEL_ID}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(DEVICE).eval()
    print(f"  Loaded in {time.time()-t0:.1f}s | {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Baseline
    baseline = generate(model, tokenizer, PROMPT)
    print(f"\n[BASELINE] \"{baseline}\"")
    
    # Extract weights
    full_sd = {}
    for k, v in model.model.state_dict().items():
        full_sd[k] = v.clone()
    products = state_dict_to_weight_products(full_sd)
    del model

    # ── Test B: SVD (no fine-tune) ──────────────────────────
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

    # ── Test B: SVD + Fine-Tune ─────────────────────────────
    print(f"\n{'─' * 60}")
    print("  TEST B-SVD-FT: SVD + Fine-Tune (200 steps)")
    print(f"{'─' * 60}")
    mapper = TransformerToTransformerMapper(projection_method="svd")
    swapped_sd, eq_map = mapper.convert(products, TARGET_B, swap_fraction=1.0)
    print(f"  Mean CKA: {eq_map.mean_cka:.4f}")
    target_model = build_target_model(TARGET_B)
    load_swapped_weights(target_model, swapped_sd)
    
    # Show output BEFORE fine-tuning
    pre_ft_output = generate(target_model, tokenizer, PROMPT)
    print(f"  Pre-FT:  \"{pre_ft_output}\"")
    
    # Fine-tune
    target_model = fine_tune_model(target_model, tokenizer, CALIBRATION_TEXTS, 
                                    num_steps=200, lr=5e-5, device=DEVICE)
    
    # Show output AFTER fine-tuning
    post_ft_output = generate(target_model, tokenizer, PROMPT)
    print(f"  Post-FT: \"{post_ft_output}\"")
    del target_model

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:    \"{baseline}\"")
    print(f"  SVD (no FT): \"{svd_output}\"")
    print(f"  SVD + FT:    \"{post_ft_output}\"")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
