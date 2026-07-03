"""
Experiment 001: Core Phase 1 proof — Transformer → Mamba weight equivalence.

Runs baselines, Paradom swap with strengthened mapping, and writes a full report.
"""

import json
import os
import sys
print(f"DEBUG: sys.path is {sys.path}")
print(f"DEBUG: CWD is {os.getcwd()}")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from paradom.models.tiny_transformer import TinyTransformer
from paradom.models.tiny_mamba import TinyMamba
from scripts.data_utils import get_wikitext_subset, CausalDataset
from paradom.core.engine import Paradom
from safetensors.torch import load_file


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return torch.exp(torch.tensor(total_loss / len(loader))).item()


def load_swapped_mamba(swapped_path: str, device) -> TinyMamba:
    swapped_weights = load_file(swapped_path)
    model = TinyMamba().to(device)
    state = model.state_dict()
    for k, v in swapped_weights.items():
        if k in state:
            state[k] = v.to(device)
    model.load_state_dict(state)
    return model


def interpret(ms_ppl, m_ppl, mr_ppl, t_ppl) -> str:
    retention = (mr_ppl - ms_ppl) / (mr_ppl - m_ppl) if mr_ppl > m_ppl else 0.0
    lines = []

    if ms_ppl < mr_ppl:
        lines.append("Swapped model beats random initialization.")
    else:
        lines.append("Swapped model does NOT beat random initialization.")

    if ms_ppl < m_ppl * 1.20:
        lines.append("Within 20% of trained Mamba baseline — **hypothesis SUPPORTED**.")
        status = "SUCCESS"
    elif ms_ppl < mr_ppl:
        lines.append(
            f"Partial transfer ({retention:.1%} retention toward trained baseline) — "
            "**hypothesis PARTIALLY SUPPORTED**."
        )
        status = "PARTIAL SUCCESS"
    else:
        lines.append("Swap failed to preserve meaningful intelligence.")
        status = "FAILURE"

    if t_ppl < ms_ppl:
        lines.append(
            f"Source Transformer ({t_ppl:.0f} ppl) remains stronger than swapped target "
            f"({ms_ppl:.0f} ppl), as expected for cross-arch transfer."
        )
    return status, retention, " ".join(lines)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("research", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    tokenized, _ = get_wikitext_subset(num_samples=1000)
    dataset = CausalDataset(tokenized)
    loader = DataLoader(dataset, batch_size=32)

    ckpt_t = "checkpoints/tiny_transformer_trained.pt"
    ckpt_m = "checkpoints/tiny_mamba_trained.pt"
    if not os.path.exists(ckpt_t) or not os.path.exists(ckpt_m):
        print("Error: Run scripts/train_poc.py first.")
        return

    transformer_weights = torch.load(ckpt_t, map_location=device)
    mamba_trained_weights = torch.load(ckpt_m, map_location=device)

    print("\n--- Evaluating Baselines ---")
    transformer = TinyTransformer().to(device)
    transformer.load_state_dict(transformer_weights)
    t_ppl = evaluate(transformer, loader, device)
    print(f"Transformer Perplexity: {t_ppl:.2f}")

    mamba_baseline = TinyMamba().to(device)
    mamba_baseline.load_state_dict(mamba_trained_weights)
    m_ppl = evaluate(mamba_baseline, loader, device)
    print(f"Mamba (Trained) Perplexity: {m_ppl:.2f}")

    mamba_random = TinyMamba().to(device)
    mr_ppl = evaluate(mamba_random, loader, device)
    print(f"Mamba (Random) Perplexity: {mr_ppl:.2f}")

    swap_fraction = 0.20  # optimal from Experiment 002 ratio sweep
    print(f"\n--- Running Paradom Swap (fraction={swap_fraction:.0%}) ---")
    engine = Paradom()
    report = engine.swap(
        source=transformer_weights,
        target_architecture="tinymamba",
        target_config={},
        swap_fraction=swap_fraction,
        output_path="./output/transformer_as_mamba",
    )

    print("\n--- Evaluating Swapped Model ---")
    mamba_swapped = load_swapped_mamba(
        "./output/transformer_as_mamba/model.safetensors", device
    )
    ms_ppl = evaluate(mamba_swapped, loader, device)
    print(f"Swapped Mamba Perplexity: {ms_ppl:.2f}")

    status, retention, interpretation = interpret(ms_ppl, m_ppl, mr_ppl, t_ppl)
    print(f"\nIntelligence Retention: {retention:.1%}")
    print(f"Mean CKA: {report.mean_cka:.3f}")
    print(f"Quality Tier: {report.quality_tier.value}")
    print(f"Layers swapped: {report.weights_swapped}")

    results = {
        "experiment": "001",
        "swap_fraction": swap_fraction,
        "perplexity": {
            "transformer_source": round(t_ppl, 2),
            "mamba_trained": round(m_ppl, 2),
            "mamba_random": round(mr_ppl, 2),
            "mamba_swapped": round(ms_ppl, 2),
        },
        "intelligence_retention": round(retention, 4),
        "mean_cka": round(report.mean_cka, 4),
        "quality_tier": report.quality_tier.value,
        "weights_swapped": report.weights_swapped,
        "swap_type_distribution": report.swap_type_distribution,
        "status": status,
    }
    with open("research/EXPERIMENT_001_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("research/EXPERIMENT_001_RESULTS.md", "w") as f:
        f.write("# Experiment 001: Transformer → Mamba Weight Equivalence\n\n")
        f.write("Phase 1 core proof using strengthened explicit mapping + SSM derivation.\n\n")
        f.write("## Perplexity Results\n\n")
        f.write("| Model | Perplexity |\n")
        f.write("|---|---:|\n")
        f.write(f"| Transformer (Source) | {t_ppl:.2f} |\n")
        f.write(f"| Mamba (Trained Baseline) | {m_ppl:.2f} |\n")
        f.write(f"| Mamba (Random Init) | {mr_ppl:.2f} |\n")
        f.write(f"| **Swapped Mamba (Paradom)** | **{ms_ppl:.2f}** |\n\n")
        f.write("## Metrics\n\n")
        f.write(f"- **Swap fraction:** {swap_fraction:.0%}\n")
        f.write(f"- **Intelligence retention:** {retention:.1%}\n")
        f.write(f"- **Mean CKA:** {report.mean_cka:.3f}\n")
        f.write(f"- **Quality tier:** {report.quality_tier.value}\n")
        f.write(f"- **Layers converted:** {report.weights_swapped}\n")
        f.write(f"- **Swap types:** {report.swap_type_distribution}\n\n")
        f.write("## Phase 1 Verdict\n\n")
        f.write(f"**{status}** — {interpretation}\n\n")
        f.write("## Mapping Used\n\n")
        f.write("| Transformer | Mamba | Method |\n")
        f.write("|---|---|---|\n")
        f.write("| q_proj + k_proj | in_proj (1st half) | Direct concat |\n")
        f.write("| v_proj | in_proj (2nd half) | Direct concat |\n")
        f.write("| o_proj | out_proj | SVD projected |\n")
        f.write("| gate + up | x_proj | SVD projected |\n")
        f.write("| down | dt_proj | SVD projected |\n")
        f.write("| input_layernorm | norm | Direct |\n")
        f.write("| Q @ K^T eigenstructure | A_log | Derived |\n")
        f.write("| v_proj energy | D, conv1d | Derived |\n")
        f.write("| embedding, lm_head | embedding, lm_head | Direct |\n")


if __name__ == "__main__":
    main()
