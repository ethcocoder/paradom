"""
AWFE Ablation Test — Find the Bottleneck Projection
=====================================================
Project all weights EXCEPT one category using original source weights
(padded/truncated to fit target shape), then measure output quality.

If replacing category X with originals fixes output → X is the bottleneck.
If no single category fixes it → the damage is distributed across all projections.

Categories tested:
  1. Embeddings      — embed_tokens.weight, lm_head.weight
  2. Q/K/V           — all q_proj, k_proj, v_proj across all layers
  3. O projections   — all o_proj across all layers
  4. FFN weights     — gate_proj, up_proj, down_proj across all layers
  5. LayerNorms      — all layernorm + final norm weights
  6. Individual layer N — every weight in layer N

Also tests individual layer slices (0, 14, 29) to find early vs late layer sensitivity.

Usage:
  python scratch/ablation_test.py          # CPU
  python scratch/ablation_test.py --gpu    # CUDA
"""
import sys, os, time, re, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from paradom.core.types import WeightProduct
from paradom.core.enums import FunctionalRole
from paradom.core.matcher import FunctionalRoleMatcher
from paradom.mappings.transformer_to_transformer import TransformerToTransformerMapper
from paradom.core.swap_engine import collect_kv_activations

# ── Configuration ────────────────────────────────────────────
MODEL_ID = "HuggingFaceTB/SmolLM-135M"
PROMPT = "Once upon a time in a land far away,"
MAX_NEW_TOK = 50

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

matcher = FunctionalRoleMatcher()

# ── Helpers ──────────────────────────────────────────────────

def state_dict_to_weight_products(sd, architecture="llama"):
    products = []
    for key, tensor in sd.items():
        role = matcher.assign_role(key, architecture)
        layer_idx = -1
        m = re.search(r"layers\.(\d+)\.", key)
        if m:
            layer_idx = int(m.group(1))
        products.append(WeightProduct(
            name=key, tensor=tensor.float(),
            shape=tuple(tensor.shape), functional_role=role,
            paradigm="transformer", architecture=architecture,
            layer_index=layer_idx, dtype=tensor.dtype,
        ))
    return products


def generate(model, tokenizer, prompt, max_new=MAX_NEW_TOK, device="cpu"):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def build_target_model(target_cfg, device="cpu"):
    cfg = LlamaConfig(
        hidden_size=target_cfg["d_model"],
        intermediate_size=target_cfg["d_inner"],
        num_hidden_layers=target_cfg["num_hidden_layers"],
        num_attention_heads=target_cfg["num_heads"],
        num_key_value_heads=target_cfg["num_key_value_heads"],
        vocab_size=target_cfg["vocab_size"],
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(cfg).to(device).eval()


def load_weights(model, swapped_sd):
    model_sd = model.model.state_dict()
    loaded = 0
    for key in model_sd:
        if key in swapped_sd:
            src = swapped_sd[key]
            if src.shape == model_sd[key].shape:
                model_sd[key] = src.to(model_sd[key].dtype)
                loaded += 1
    model.model.load_state_dict(model_sd, strict=False)
    if hasattr(model, 'lm_head') and hasattr(model.model, 'embed_tokens'):
        model.lm_head.weight = model.model.embed_tokens.weight
    return loaded


def truncate_to_target(src_tensor, target_shape):
    """Fit a source tensor into target shape by truncation (and zero-padding if needed)."""
    result = torch.zeros(target_shape, dtype=torch.float32)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(src_tensor.shape, target_shape))
    result[slices] = src_tensor[slices].float()
    return result


def classify_key(key):
    """Classify a state_dict key into an ablation category."""
    if "embed_tokens" in key or "lm_head" in key:
        return "embeddings"
    if "self_attn.q_proj" in key or "self_attn.k_proj" in key or "self_attn.v_proj" in key:
        return "qkv"
    if "self_attn.o_proj" in key:
        return "o_proj"
    if "mlp.gate_proj" in key or "mlp.up_proj" in key or "mlp.down_proj" in key:
        return "ffn"
    if "layernorm" in key or "norm." in key:
        return "norms"
    return "other"


def make_ablation(full_swapped, source_sd, target_config, category, layer_idx=None):
    """Create a variant where `category` uses original source weights (truncated/padded)."""
    variant = {}
    for key, proj_tensor in full_swapped.items():
        use_original = False
        if category == "embeddings":
            use_original = classify_key(key) == "embeddings"
        elif category == "qkv":
            use_original = classify_key(key) == "qkv"
        elif category == "o_proj":
            use_original = classify_key(key) == "o_proj"
        elif category == "ffn":
            use_original = classify_key(key) == "ffn"
        elif category == "norms":
            use_original = classify_key(key) == "norms"
        elif category.startswith("layer_"):
            target_layer = int(category.split("_")[1])
            m = re.search(r"layers\.(\d+)\.", key)
            if m and int(m.group(1)) == target_layer:
                use_original = True

        if use_original:
            if key in source_sd:
                variant[key] = truncate_to_target(source_sd[key], proj_tensor.shape)
            else:
                variant[key] = proj_tensor
        else:
            variant[key] = proj_tensor
    return variant


def eval_variant(model, tokenizer, variant_sd, device, variant_name, baseline_text):
    """Load variant, generate text, return output and token match count."""
    load_weights(model, variant_sd)
    output = generate(model, tokenizer, PROMPT, device=device)

    # Token overlap with baseline as a rough quality metric
    baseline_tokens = baseline_text.split()
    output_tokens = output.split()
    if len(baseline_tokens) == 0:
        match_frac = 0.0
    else:
        overlap = sum(1 for t in output_tokens if t in baseline_tokens)
        match_frac = overlap / len(baseline_tokens)

    return output, match_frac


# ── MAIN ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  AWFE Ablation Test — Bottleneck Projection Finder")
    print("=" * 70)
    print(f"  Device: {device}")

    # ── Load source model ──
    print(f"\n[1/4] Loading {MODEL_ID}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(device).eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Baseline output ──
    baseline = generate(model, tokenizer, PROMPT, device=device)
    print(f"\n[BASELINE] \"{baseline}\"")

    # ── Extract source weights ──
    source_sd = {}
    for k, v in model.model.state_dict().items():
        source_sd[k] = v.clone()
    if hasattr(model, 'lm_head'):
        source_sd['lm_head.weight'] = model.lm_head.weight.data.clone()
    products = state_dict_to_weight_products(source_sd)
    print(f"  {len(products)} source tensors")

    # ── Collect KV activations ──
    print("  Collecting KV activations...")
    kv_acts = collect_kv_activations(model, tokenizer, PROMPT)
    print(f"  Collected from {len(kv_acts)} layers")
    
    # ── Calibrate activation-aware projector ──
    from paradom.core.activation_aware_projector import ActivationAwareProjector
    print("  Calibrating activation-aware projector...")
    projector = ActivationAwareProjector(SOURCE_CONFIG, TARGET_B)
    projector.calibrate(model, tokenizer, PROMPT)
    print("  Projector calibrated")

    # Free source model memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Full swap (Test B config) ──
    print(f"\n[2/4] Running full swap (576d→512d)...")
    mapper = TransformerToTransformerMapper(force_projected=False, source_config=SOURCE_CONFIG)
    mapper.set_kv_activations(kv_acts)
    t0 = time.time()
    full_swapped, eq_map = mapper.convert(products, TARGET_B, swap_fraction=1.0)
    print(f"  Swap done in {time.time()-t0:.1f}s | {len(full_swapped)} tensors | CKA: {eq_map.mean_cka:.4f}")
    
    # Also run with activation-aware projector
    print(f"\n  Running with activation-aware projector...")
    mapper_proj = TransformerToTransformerMapper(force_projected=False, source_config=SOURCE_CONFIG)
    mapper_proj.set_kv_activations(kv_acts)
    mapper_proj.set_projector(projector)
    full_swapped_proj, eq_map_proj = mapper_proj.convert(products, TARGET_B, swap_fraction=1.0)
    print(f"  Projector swap done | CKA: {eq_map_proj.mean_cka:.4f}")

    # ── Full swap output ──
    print("\n[3/4] Evaluating full swap, projector swap, and ablations...")
    target_model = build_target_model(TARGET_B, device=device)
    load_weights(target_model, full_swapped)
    full_output = generate(target_model, tokenizer, PROMPT, device=device)
    _, full_match = eval_variant(target_model, tokenizer, full_swapped, device, "full_swap", baseline)
    print(f"\n  [FULL SWAP]  \"{full_output}\"")
    print(f"  Token overlap with baseline: {full_match:.2%}")
    
    # Evaluate projector swap
    load_weights(target_model, full_swapped_proj)
    proj_output = generate(target_model, tokenizer, PROMPT, device=device)
    _, proj_match = eval_variant(target_model, tokenizer, full_swapped_proj, device, "projector", baseline)
    print(f"\n  [PROJECTOR]  \"{proj_output}\"")
    print(f"  Token overlap with baseline: {proj_match:.2%}")

    # ── Ablation categories ──
    categories = [
        "embeddings", "qkv", "o_proj", "ffn", "norms",
    ] + [f"layer_{i}" for i in [0, 1, 2, 14, 15, 28, 29]]

    results = []
    results.append(("(baseline)", baseline, 1.0))
    results.append(("(full_swap)", full_output, full_match))
    results.append(("(projector)", proj_output, proj_match))

    t_start = time.time()
    for cat in categories:
        variant = make_ablation(full_swapped, source_sd, TARGET_B, cat)
        out, match = eval_variant(target_model, tokenizer, variant, device, cat, baseline)
        results.append((cat, out, match))
        print(f"  [{cat:20s}] overlap={match:.2%}  \"{out[:80]}...\"")
        # Reload full swap for next ablation
        load_weights(target_model, full_swapped)

    elapsed = time.time() - t_start
    print(f"\n  All ablations done in {elapsed:.1f}s")

    # ── Print comparison table ──
    print(f"\n[4/4] Results")
    print("=" * 70)
    print(f"{'Category':<20} {'Overlap':>8}  {'Δ vs full':>8}  Output (first 100 chars)")
    print("-" * 70)
    baseline_overlap = 1.0
    full_overlap = full_match
    for name, out, match in results:
        delta = match - full_overlap
        marker = ""
        if delta > 0.05:
            marker = " **"
        elif delta > 0.0:
            marker = " *"
        print(f"{name:<20} {match:>7.1%}  {delta:>+7.1%}  {out[:100]}{marker}")
    print("-" * 70)
    print("  ** = replacing this category with originals improves output significantly")
    print("  *  = replacing this category with originals improves output slightly")
    print()

    # ── Identify bottleneck ──
    ablation_results = [(n, m) for n, _, m in results if n not in ("(baseline)", "(full_swap)")]
    best = max(ablation_results, key=lambda x: x[1])
    print(f"  BEST ABLATION: {best[0]} (overlap={best[1]:.2%})")
    if best[1] - full_match > 0.05:
        print(f"  >> BOTTLENECK: Replacing '{best[0]}' with source originals fixes output.")
        print(f"     The SVD projection of this category is the primary damage source.")
    elif best[1] - full_match > 0.0:
        print(f"  >> MILD BENEFIT: Replacing '{best[0]}' helps slightly.")
        print(f"     Damage is distributed across multiple projection categories.")
    else:
        print(f"  >> NO SINGLE BOTTLENECK: No category replacement helps.")
        print(f"     The damage comes from cross-category interaction,")
        print(f"     not any individual projection.")

    # Cleanup
    del target_model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"\n{'=' * 70}")
    print("  ABLATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
