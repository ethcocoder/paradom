"""Aggregate development-only Experiment 016 gate-boundary diagnostics."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("research/experiment_016_interface_boundary")
OUT = Path("research/experiment_016_analysis")
SEEDS = (20260841, 20260842, 20260843, 20260844, 20260845)
ALPHAS = ("0.00", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00")


def mean_sd(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "minimum": min(values), "maximum": max(values)}


def fmt(value):
    return f"{value:.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = []
    for seed in SEEDS:
        path = ROOT / f"seed_{seed}" / "results.json"
        result = json.loads(path.read_text())
        if result["data_guardrail"]["test_split_loaded"]:
            raise RuntimeError(f"Test-split guardrail failed for seed {seed}")
        raw.append(result)

    metrics_full = list(raw[0]["alpha_sweep"]["0.80"]["full_48"].keys())
    metrics_short = list(raw[0]["alpha_sweep"]["0.80"]["short_24"].keys())
    curves = {alpha: {"full_48": {}, "short_24": {}} for alpha in ALPHAS}
    for alpha in ALPHAS:
        for metric in metrics_full:
            curves[alpha]["full_48"][metric] = mean_sd([record["alpha_sweep"][alpha]["full_48"][metric] for record in raw])
        for metric in metrics_short:
            curves[alpha]["short_24"][metric] = mean_sd([record["alpha_sweep"][alpha]["short_24"][metric] for record in raw])

    boundary = {}
    for metric in metrics_full:
        differences = [record["alpha_sweep"]["0.90"]["full_48"][metric] - record["alpha_sweep"]["0.80"]["full_48"][metric] for record in raw]
        boundary[f"full_48_{metric}_090_minus_080"] = mean_sd(differences)
    for metric in metrics_short:
        differences = [record["alpha_sweep"]["0.90"]["short_24"][metric] - record["alpha_sweep"]["0.80"]["short_24"][metric] for record in raw]
        boundary[f"short_24_{metric}_090_minus_080"] = mean_sd(differences)

    # Compare the 0.80→0.90 increment with the preceding 0.70→0.80 increment.
    slope_ratios = {}
    for metric in ("ce", "logit_kl", "post_layer1_relative_l2_from_080", "post_layer2_relative_l2_from_080"):
        increments_7080 = [record["alpha_sweep"]["0.80"]["full_48"][metric] - record["alpha_sweep"]["0.70"]["full_48"][metric] for record in raw]
        increments_8090 = [record["alpha_sweep"]["0.90"]["full_48"][metric] - record["alpha_sweep"]["0.80"]["full_48"][metric] for record in raw]
        ratios = [after / before for before, after in zip(increments_7080, increments_8090) if abs(before) > 1e-12]
        slope_ratios[metric] = {"increment_070_to_080": mean_sd(increments_7080), "increment_080_to_090": mean_sd(increments_8090), "ratio_080_090_over_070_080": mean_sd(ratios)}

    controller = []
    for result in raw:
        l1 = result["reconstructed_stages"]["layer_1"]["gate_result"]["controller"]
        controller.append({"seed": result["seed"], "final_alpha": result["reconstructed_stages"]["layer_1"]["gate_result"]["final_alpha"], "target_080": next(x for x in l1 if x["target"] == 0.8), "target_090": next(x for x in l1 if x["target"] == 0.9), "target_100": next(x for x in l1 if x["target"] == 1.0)})

    per_seed = []
    for result in raw:
        sweep = result["alpha_sweep"]
        per_seed.append({
            "seed": result["seed"],
            "layer1_final_alpha": result["reconstructed_stages"]["layer_1"]["gate_result"]["final_alpha"],
            "full_48_080": sweep["0.80"]["full_48"],
            "full_48_090": sweep["0.90"]["full_48"],
            "short_24_080": sweep["0.80"]["short_24"],
            "short_24_090": sweep["0.90"]["short_24"],
        })

    aggregate = {
        "experiment": "experiment_016_interface_boundary",
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "data_guardrail": {"test_split_loaded_any_seed": False, "test_metrics_available": False, "statement": "Only WikiText train and validation partitions were requested; no final-test metric was calculated."},
        "curves": curves,
        "boundary_090_minus_080": boundary,
        "slope_ratios": slope_ratios,
        "controller": controller,
        "per_seed_boundary": per_seed,
        "diagnostic_conclusions": {
            "reconstructed_layer1_alpha_080_all_seeds": all(row["layer1_final_alpha"] == 0.8 for row in per_seed),
            "kl_exceeds_8_at_alpha_090_all_seeds": all(row["full_48_090"]["logit_kl"] > 8.0 for row in per_seed),
            "directional_or_value_metrics_prove_global_compatibility": False,
            "test_set_claim_permitted": False,
        },
    }
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = ["# Experiment 016: Development-Only Gate-Boundary Aggregate", "", "## Guardrail", "", "Only WikiText-2 train and validation partitions were requested. The frozen test split was not loaded or scored. All numbers below are development diagnostics, not generalization results.", "", "## Mean Development Curves", "", "| Alpha | CE (48) | KL (48) | CE (24) | KL (24) | Post-L1 drift from 0.80 | Post-L2 drift from 0.80 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for alpha in ALPHAS:
        full, short = curves[alpha]["full_48"], curves[alpha]["short_24"]
        lines.append(f"| {alpha} | {fmt(full['ce']['mean'])} ± {fmt(full['ce']['sample_std'])} | {fmt(full['logit_kl']['mean'])} ± {fmt(full['logit_kl']['sample_std'])} | {fmt(short['ce']['mean'])} ± {fmt(short['ce']['sample_std'])} | {fmt(short['logit_kl']['mean'])} ± {fmt(short['logit_kl']['sample_std'])} | {fmt(full['post_layer1_relative_l2_from_080']['mean'])} | {fmt(full['post_layer2_relative_l2_from_080']['mean'])} |")
    lines += ["", "## Boundary Delta: 0.90 − 0.80", "", "| Metric | Mean ± sample SD |", "|---|---:|"]
    for key in ("full_48_ce_090_minus_080", "full_48_logit_kl_090_minus_080", "short_24_ce_090_minus_080", "short_24_logit_kl_090_minus_080", "full_48_post_layer1_relative_l2_from_080_090_minus_080", "full_48_post_layer2_relative_l2_from_080_090_minus_080", "full_48_branch_rms_ratio_090_minus_080", "full_48_branch_feature_log_variance_ratio_abs_090_minus_080", "full_48_branch_token_gram_relative_l2_090_minus_080"):
        item = boundary[key]
        lines.append(f"| {key.replace('_', ' ')} | {fmt(item['mean'])} ± {fmt(item['sample_std'])} |")
    lines += ["", "## Controller Reproduction", "", "| Seed | Accepted L1 alpha | KL at 0.80 | KL at 0.90 | 0.90 accepted? |", "|---:|---:|---:|---:|---|"]
    for row in controller:
        lines.append(f"| {row['seed']} | {row['final_alpha']:.2f} | {fmt(row['target_080']['development_kl'])} | {fmt(row['target_090']['development_kl'])} | {row['target_090']['accepted']} |")
    (OUT / "aggregate_results.md").write_text("\n".join(lines) + "\n")

    xs = [float(alpha) for alpha in ALPHAS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(xs, [curves[a]["full_48"]["logit_kl"]["mean"] for a in ALPHAS], marker="o", label="48 tokens")
    axes[0, 0].plot(xs, [curves[a]["short_24"]["logit_kl"]["mean"] for a in ALPHAS], marker="s", label="24 tokens")
    axes[0, 0].axhline(8.0, color="#BF616A", linestyle="--", label="Controller KL threshold")
    axes[0, 0].axvline(.8, color="#4C566A", linestyle=":"); axes[0, 0].set(title="Source-logit KL rises smoothly through boundary", xlabel="Layer-1 gate alpha", ylabel="Development KL"); axes[0, 0].legend()
    axes[0, 1].plot(xs, [curves[a]["full_48"]["ce"]["mean"] for a in ALPHAS], marker="o", label="48 tokens")
    axes[0, 1].plot(xs, [curves[a]["short_24"]["ce"]["mean"] for a in ALPHAS], marker="s", label="24 tokens")
    axes[0, 1].axvline(.8, color="#4C566A", linestyle=":"); axes[0, 1].set(title="Development CE versus gate alpha", xlabel="Layer-1 gate alpha", ylabel="Token CE"); axes[0, 1].legend()
    axes[1, 0].plot(xs, [curves[a]["full_48"]["post_layer1_relative_l2_from_080"]["mean"] for a in ALPHAS], marker="o", label="Post layer 1")
    axes[1, 0].plot(xs, [curves[a]["full_48"]["post_layer2_relative_l2_from_080"]["mean"] for a in ALPHAS], marker="s", label="Post layer 2")
    axes[1, 0].axvline(.8, color="#4C566A", linestyle=":"); axes[1, 0].set(title="Residual drift from alpha=0.80", xlabel="Layer-1 gate alpha", ylabel="Relative L2"); axes[1, 0].legend()
    labels = ["RMS ratio", "Feature log-var", "Token Gram"]
    vals = [curves["0.80"]["full_48"]["branch_rms_ratio"]["mean"], curves["0.80"]["full_48"]["branch_feature_log_variance_ratio_abs"]["mean"], curves["0.80"]["full_48"]["branch_token_gram_relative_l2"]["mean"]]
    axes[1, 1].bar(labels, vals, color=["#5E81AC", "#D08770", "#A3BE8C"])
    axes[1, 1].set(title="Static branch mismatch at alpha=0.80", ylabel="Diagnostic magnitude")
    fig.savefig(OUT / "boundary_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"boundary": boundary, "slope_ratios": slope_ratios, "conclusions": aggregate["diagnostic_conclusions"]}, indent=2))

if __name__ == "__main__": main()
