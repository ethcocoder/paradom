"""
AWFE Phase 2 — Ultimate Proof of Concept
=========================================
THREE TESTS to prove Weight Force Equivalence:

Test A: SVD Projection (same shape) → decompose every weight via SVD and
        reconstruct. If the model still talks, SVD preserves intelligence.

Test B: Cross-Dimension Swap → project SmolLM-135M (576 hidden) weights
        into a SMALLER architecture (512 hidden, 8 heads). This is the
        TRUE AWFE test — different shape, different architecture.

Test C: Cross-Dimension Swap → project into a LARGER architecture
        (768 hidden, 12 heads). Upscaling test.
"""

import sys, os, time, re, copy
import torch

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

# SmolLM-135M source config
SOURCE_CONFIG = {
    "d_model": 576, "d_inner": 1536,
    "num_heads": 9, "num_key_value_heads": 3,
    "head_dim": 64, "num_hidden_layers": 30,
    "vocab_size": 49152,
}

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
    """Create a fresh LlamaForCausalLM with custom config."""
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
    """Load swapped weights into HF model, report stats."""
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
    # Handle tied embeddings
    if hasattr(model, 'lm_head') and hasattr(model.model, 'embed_tokens'):
        model.lm_head.weight = model.model.embed_tokens.weight
    
    return loaded, skipped, mismatched


def run_test(test_name, source_products, target_config, tokenizer, force_projected=False):
    """Run a single swap test and return the generated text."""
    print(f"\n{'─' * 60}")
    print(f"  {test_name}")
    print(f"{'─' * 60}")
    
    # Build mapper
    mapper = TransformerToTransformerMapper(force_projected=force_projected)
    
    # Swap
    t0 = time.time()
    swapped_sd, eq_map = mapper.convert(source_products, target_config, swap_fraction=1.0)
    elapsed = time.time() - t0
    print(f"  Swap: {elapsed:.1f}s | {len(swapped_sd)} tensors | Mean CKA: {eq_map.mean_cka:.4f}")
    
    # Build target model shell
    target_model = build_target_model(target_config)
    loaded, skipped, mismatched = load_swapped_weights(target_model, swapped_sd)
    print(f"  Loaded: {loaded} | Skipped: {skipped} | Mismatch: {mismatched}")
    
    # Generate
    output = generate(target_model, tokenizer, PROMPT)
    print(f"  Output: \"{output}\"")
    
    # Cleanup
    del target_model
    return output, eq_map.mean_cka


# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AWFE End-to-End Intelligence Proof")
    print("=" * 60)

    # Load original model
    print(f"\n[LOAD] {MODEL_ID}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
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
    
    roles = {}
    for wp in products:
        roles[wp.functional_role.value] = roles.get(wp.functional_role.value, 0) + 1
    print(f"  {len(products)} tensors | Roles: {roles}")
    
    # Free original model memory
    del model

    # ── Test A: SVD Projection same-shape ────────────────────
    run_test(
        "TEST A: SVD Decompose → Reconstruct (same shape, force_projected)",
        products, SOURCE_CONFIG, tokenizer, force_projected=True
    )

    # ── Test B: Cross-dim DOWNSCALE ──────────────────────────
    target_b = {
        "d_model": 512, "d_inner": 1408,
        "num_heads": 8, "num_key_value_heads": 2,
        "head_dim": 64, "num_hidden_layers": 30,
        "vocab_size": 49152,
    }
    run_test(
        "TEST B: Cross-Dimension DOWNSCALE (576→512 hidden)",
        products, target_b, tokenizer
    )

    # ── Test C: Cross-dim UPSCALE ────────────────────────────
    target_c = {
        "d_model": 768, "d_inner": 2048,
        "num_heads": 12, "num_key_value_heads": 4,
        "head_dim": 64, "num_hidden_layers": 30,
        "vocab_size": 49152,
    }
    run_test(
        "TEST C: Cross-Dimension UPSCALE (576→768 hidden)",
        products, target_c, tokenizer
    )

    print(f"\n{'=' * 60}")
    print("  ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
