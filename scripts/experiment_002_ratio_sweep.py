"""Swap fraction sensitivity analysis for Phase 1."""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


def load_swapped(path: str, device) -> TinyMamba:
    weights = load_file(path)
    model = TinyMamba().to(device)
    state = model.state_dict()
    for k, v in weights.items():
        if k in state:
            state[k] = v.to(device)
    model.load_state_dict(state)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("research", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    ckpt_t = "checkpoints/tiny_transformer_trained.pt"
    ckpt_m = "checkpoints/tiny_mamba_trained.pt"
    if not os.path.exists(ckpt_t) or not os.path.exists(ckpt_m):
        print("Error: Run scripts/train_poc.py first.")
        return

    tokenized, _ = get_wikitext_subset(num_samples=500)
    loader = DataLoader(CausalDataset(tokenized), batch_size=16)

    transformer_weights = torch.load(ckpt_t, map_location=device)
    mamba_trained = torch.load(ckpt_m, map_location=device)

    mamba_baseline = TinyMamba().to(device)
    mamba_baseline.load_state_dict(mamba_trained)
    m_ppl = evaluate(mamba_baseline, loader, device)

    mamba_random = TinyMamba().to(device)
    mr_ppl = evaluate(mamba_random, loader, device)

    ratios = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
    engine = Paradom()
    results = []

    print(f"Mamba trained: {m_ppl:.2f} | Random: {mr_ppl:.2f}\n")

    for fraction in ratios:
        out_dir = f"./output/sweep_ratio_{int(fraction * 100)}"
        engine.swap(
            source=transformer_weights,
            target_architecture="tinymamba",
            target_config={},
            swap_fraction=fraction,
            output_path=out_dir,
        )
        ppl = evaluate(load_swapped(f"{out_dir}/model.safetensors", device), loader, device)
        retention = (mr_ppl - ppl) / (mr_ppl - m_ppl) if mr_ppl > m_ppl else 0.0
        within_20 = ppl < m_ppl * 1.20
        results.append({
            "ratio": fraction,
            "perplexity": round(ppl, 2),
            "retention": round(retention, 4),
            "within_20pct_of_trained": within_20,
        })
        print(f"  {fraction:.0%}: ppl={ppl:.2f} retention={retention:.1%} supported={within_20}")

    best = min(results, key=lambda r: r["perplexity"])
    with open("research/EXPERIMENT_002_RATIO_SWEEP.json", "w") as f:
        json.dump({"baseline_trained": m_ppl, "baseline_random": mr_ppl, "sweep": results, "best": best}, f, indent=2)

    with open("research/EXPERIMENT_002_RATIO_SWEEP.md", "w") as f:
        f.write("# Experiment 002: Swap Ratio Sensitivity Analysis\n\n")
        f.write("Strengthened Phase 1 mapping with SSM derivation.\n\n")
        f.write("| Swap Fraction | Perplexity | Retention | Within 20% of Trained? |\n")
        f.write("|---:|---:|---:|:---:|\n")
        for r in results:
            mark = "Yes" if r["within_20pct_of_trained"] else "No"
            f.write(f"| {r['ratio']:.0%} | {r['perplexity']:.2f} | {r['retention']:.1%} | {mark} |\n")
        f.write(f"\n**Best fraction:** {best['ratio']:.0%} (ppl={best['perplexity']:.2f})\n")
        f.write(f"**Trained baseline:** {m_ppl:.2f}\n")
        f.write(f"**Random baseline:** {mr_ppl:.2f}\n")

    print(f"\nBest ratio: {best['ratio']:.0%} → ppl {best['perplexity']:.2f}")


if __name__ == "__main__":
    main()
