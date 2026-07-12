"""
AWFE Phase 2 — Functional Projection Test
==========================================
Tests functional importance profiling vs SVD for downscale projection.

Key insight from compressor analysis:
- SVD finds mathematically optimal directions (for reconstruction)
- Functional projection finds directions the model actually uses (for generation)
"""

import sys, os, time, re, copy
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from paradom.core.types import WeightProduct
from paradom.core.enums import FunctionalRole
from paradom.core.matcher import FunctionalRoleMatcher
from paradom.core.functional_importance import FunctionalImportanceProfiler
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

# Test B target config
TARGET_B = {
    "d_model": 512, "d_inner": 1408,
    "num_heads": 8, "num_key_value_heads": 2,
    "head_dim": 64, "num_hidden_layers": 30,
    "vocab_size": 49152,
}

# ── Helpers ──────────────────────────────────────────────────
matcher = FunctionalRoleMatcher()

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
]

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


def collect_calibration_data(model, tokenizer, calibration_texts, max_length=256, device="cpu"):
    """
    Collect hidden states from the last layer of the source model.
    Returns tensor of shape (n_tokens, d_model).
    """
    model.eval()
    model.to(device)
    
    hidden_states = []
    
    # Hook to capture last layer hidden states
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach().cpu())
    
    # Register hook on last layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        last_layer = model.model.layers[-1]
        handle = last_layer.register_forward_hook(hook_fn)
        
        # Run calibration
        print(f"  [Calibration] Collecting hidden states from {len(calibration_texts)} texts...")
        with torch.no_grad():
            for text in calibration_texts:
                inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                                   truncation=True, padding=False)
                input_ids = inputs.input_ids.to(device)
                hidden_states.clear()
                model(input_ids)
        
        handle.remove()
    
    if hidden_states:
        all_hidden = torch.cat(hidden_states, dim=0)  # (n_tokens, d_model)
        print(f"  [Calibration] Collected {all_hidden.shape[0]} tokens, d_model={all_hidden.shape[1]}")
        return all_hidden
    else:
        print("  [Calibration] WARNING: No hidden states collected")
        return None


def run_test(test_name, source_products, target_config, tokenizer, 
             projection_method="svd", calibration_data=None):
    """Run a single swap test and return the generated text."""
    print(f"\n{'─' * 60}")
    print(f"  {test_name}")
    print(f"{'─' * 60}")
    
    # Build mapper
    mapper = TransformerToTransformerMapper(
        force_projected=False,
        projection_method=projection_method
    )
    
    # Swap
    t0 = time.time()
    swapped_sd, eq_map = mapper.convert(
        source_products, target_config, 
        swap_fraction=1.0,
        calibration_data=calibration_data
    )
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
    print("  AWFE Functional Projection Test")
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
    
    # Collect calibration data (before freeing the model!)
    print(f"\n[CALIBRATION] Collecting hidden states...")
    calibration_data = collect_calibration_data(model, tokenizer, CALIBRATION_TEXTS, device=DEVICE)
    
    # Free original model memory
    del model

    # ── Test B: SVD Projection (baseline) ───────────────────
    print("\n" + "=" * 60)
    print("  COMPARING SVD vs FUNCTIONAL PROJECTION")
    print("=" * 60)
    
    svd_output, svd_cka = run_test(
        "TEST B-SVD: Cross-Dimension DOWNSCALE (576→512) with SVD",
        products, TARGET_B, tokenizer,
        projection_method="svd"
    )

    # ── Test B: Functional Projection ────────────────────────
    functional_output, functional_cka = run_test(
        "TEST B-FUNC: Cross-Dimension DOWNSCALE (576→512) with Functional",
        products, TARGET_B, tokenizer,
        projection_method="functional",
        calibration_data=calibration_data
    )

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Baseline:  \"{baseline}\"")
    print(f"  SVD:       \"{svd_output}\"")
    print(f"  Functional:\"{functional_output}\"")
    print(f"  SVD CKA:       {svd_cka:.4f}")
    print(f"  Functional CKA:{functional_cka:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
