"""Aggregate Experiment 015 faithfully, treating adaptive endpoint failures as failures rather than imputing final losses."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("research/experiment_015_adaptive_interface")
OUT = Path("research/experiment_015_analysis")
SEEDS = (20260841, 20260842, 20260843, 20260844, 20260845)
SCIENCE = "The purpose of scientific research is to"
EXPLORER = "Once upon a time, a young explorer discovered"


def summary(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "minimum": min(values), "maximum": max(values)}


def f(value):
    return f"{value:.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, generations = [], []
    for seed in SEEDS:
        result = json.loads((ROOT / f"seed_{seed}" / "results.json").read_text())
        ce, value, adaptive, teacher = result["ce_adapter"], result["value_functional"], result["adaptive_interface"], result["teacher"]
        l0 = adaptive["stages_development"]["layer_0"]["gate_result"]
        l1 = adaptive["stages_development"]["layer_1"]["gate_result"]
        ce_loss = ce["final_test"]["loss"]
        value_loss = value["final_test"]["loss"]
        rows.append({
            "seed": seed,
            "teacher_loss": teacher["loss"],
            "ce_adapter_test_loss": ce_loss,
            "value_functional_test_loss": value_loss,
            "ce_minus_value_loss": ce_loss - value_loss,
            "adaptive_endpoint_alpha_one": adaptive["endpoint_alpha_one"],
            "adaptive_layer0_final_alpha": l0["final_alpha"],
            "adaptive_layer1_final_alpha": l1["final_alpha"],
            "adaptive_layer0_controller": l0["controller"],
            "adaptive_layer1_controller": l1["controller"],
            "adaptive_final_test_available": adaptive["final_test"] is not None,
            "ce_alpha_zero_error": abs(ce["alpha_zero_test"]["loss"] - teacher["loss"]),
            "value_alpha_zero_error": abs(value["alpha_zero_test"]["loss"] - teacher["loss"]),
            "adaptive_alpha_zero_error": abs(adaptive["alpha_zero_test"]["loss"] - teacher["loss"]),
            "value_layer1_tangent_nmse": value["stages_development"]["layer_1"]["diagnostic_after"]["tangent_nmse"],
            "adaptive_layer1_tangent_nmse": adaptive["stages_development"]["layer_1"]["diagnostic_after"]["tangent_nmse"],
        })
        generations.append({"seed": seed, "ce_science": ce["generation"][SCIENCE], "value_science": value["generation"][SCIENCE], "ce_explorer": ce["generation"][EXPLORER], "value_explorer": value["generation"][EXPLORER]})

    ce_losses = [r["ce_adapter_test_loss"] for r in rows]
    value_losses = [r["value_functional_test_loss"] for r in rows]
    paired = [r["ce_minus_value_loss"] for r in rows]
    a_tangent = [r["adaptive_layer1_tangent_nmse"] for r in rows]
    v_tangent = [r["value_layer1_tangent_nmse"] for r in rows]
    aggregate = {
        "experiment": "experiment_015_adaptive_interface",
        "seeds": list(SEEDS),
        "n": len(SEEDS),
        "endpoint_rule": "Adaptive final-test metrics are intentionally unavailable when either gate does not reach alpha=1; these are protocol failures, not missing values to be imputed.",
        "metrics": {"teacher_test_loss": summary([r["teacher_loss"] for r in rows]), "ce_adapter_test_loss": summary(ce_losses), "value_functional_test_loss": summary(value_losses), "paired_ce_minus_value_test_loss": summary(paired), "value_layer1_tangent_nmse": summary(v_tangent), "adaptive_layer1_tangent_nmse": summary(a_tangent)},
        "acceptance": {
            "alpha_zero_exact_all_conditions": all(max(r["ce_alpha_zero_error"], r["value_alpha_zero_error"], r["adaptive_alpha_zero_error"]) < 1e-5 for r in rows),
            "adaptive_both_layers_alpha_one_count": sum(r["adaptive_endpoint_alpha_one"] for r in rows),
            "adaptive_endpoint_requirement_at_least_4_of_5": sum(r["adaptive_endpoint_alpha_one"] for r in rows) >= 4,
            "adaptive_final_test_losses_available": sum(r["adaptive_final_test_available"] for r in rows),
            "ce_beats_value_count": sum(r["ce_minus_value_loss"] < 0 for r in rows),
            "adaptive_directional_diagnostic_better_than_value_count": sum(r["adaptive_layer1_tangent_nmse"] < r["value_layer1_tangent_nmse"] for r in rows),
            "third_layer_permitted": False,
        },
        "per_seed": rows,
        "generations": generations,
    }
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = ["# Experiment 015: Aggregate Results", "", "## Endpoint and Held-Out Test Results", "", "| Seed | CE-adapter test loss | Value-functional test loss | CE − value | Adaptive L0 alpha | Adaptive L1 alpha | Adaptive endpoint |", "|---:|---:|---:|---:|---:|---:|---|" ]
    for r in rows:
        lines.append(f"| {r['seed']} | {f(r['ce_adapter_test_loss'])} | {f(r['value_functional_test_loss'])} | {f(r['ce_minus_value_loss'])} | {r['adaptive_layer0_final_alpha']:.2f} | {r['adaptive_layer1_final_alpha']:.2f} | {'pass' if r['adaptive_endpoint_alpha_one'] else 'failed'} |")
    m = aggregate["metrics"]
    lines += ["", "## Aggregate", "", "| Metric | Mean ± sample SD |", "|---|---:|", f"| CE-adapter test loss | {f(m['ce_adapter_test_loss']['mean'])} ± {f(m['ce_adapter_test_loss']['sample_std'])} |", f"| Value-functional test loss | {f(m['value_functional_test_loss']['mean'])} ± {f(m['value_functional_test_loss']['sample_std'])} |", f"| CE − value test loss | {f(m['paired_ce_minus_value_test_loss']['mean'])} ± {f(m['paired_ce_minus_value_test_loss']['sample_std'])} |", f"| Value-functional L1 directional NMSE | {f(m['value_layer1_tangent_nmse']['mean'])} ± {f(m['value_layer1_tangent_nmse']['sample_std'])} |", f"| Adaptive-interface L1 directional NMSE | {f(m['adaptive_layer1_tangent_nmse']['mean'])} ± {f(m['adaptive_layer1_tangent_nmse']['sample_std'])} |", "", "Adaptive runs did not receive final test metrics because the locked endpoint rule was not met. This preserves the test protocol and prevents an alpha=0.8 partial hybrid from being presented as a two-layer replacement.", "", "## Fixed-Prompt Continuations at Valid alpha=1 Endpoints", "", "| Seed | CE-adapter science | Value-functional science |", "|---:|---|---|" ]
    for g in generations:
        lines.append(f"| {g['seed']} | {g['ce_science'].replace('|', '\\|')} | {g['value_science'].replace('|', '\\|')} |")
    (OUT / "aggregate_results.md").write_text("\n".join(lines) + "\n")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), constrained_layout=True)
    xs = list(range(len(SEEDS)))
    for i, r in enumerate(rows): axes[0].plot([i - .12, i + .12], [r["ce_adapter_test_loss"], r["value_functional_test_loss"]], color="#A3BE8C", zorder=1)
    axes[0].scatter([i - .12 for i in xs], ce_losses, color="#D08770", label="CE-adapter", s=60, zorder=2)
    axes[0].scatter([i + .12 for i in xs], value_losses, color="#5E81AC", label="Value-functional", s=60, zorder=2)
    axes[0].axhline(rows[0]["teacher_loss"], color="#4C566A", ls="--", label="Frozen teacher")
    axes[0].set_xticks(xs, [str(s) for s in SEEDS]); axes[0].set_ylabel("Final held-out loss (lower is better)"); axes[0].set_title("Valid alpha=1 endpoints"); axes[0].legend()
    axes[1].bar([i - .16 for i in xs], [r["adaptive_layer0_final_alpha"] for r in rows], .32, label="Adaptive layer 0", color="#5E81AC")
    axes[1].bar([i + .16 for i in xs], [r["adaptive_layer1_final_alpha"] for r in rows], .32, label="Adaptive layer 1", color="#BF616A")
    axes[1].axhline(1.0, color="#4C566A", ls="--", label="Required endpoint")
    axes[1].set_xticks(xs, [str(s) for s in SEEDS]); axes[1].set_ylim(0, 1.1); axes[1].set_ylabel("Accepted gate alpha"); axes[1].set_title("Adaptive controller endpoint"); axes[1].legend()
    fig.savefig(OUT / "endpoint_and_test_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"acceptance": aggregate["acceptance"], "metrics": aggregate["metrics"]}, indent=2))

if __name__ == "__main__": main()
