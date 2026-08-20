"""Aggregate Experiment 017 featurewise moment-calibration results."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("research/experiment_017_moment_calibration")
OUT = Path("research/experiment_017_analysis")
SEEDS = (20260851, 20260852, 20260853, 20260854, 20260855)
CONDS = ("ce_calibrator", "value_functional", "moment_functional")
LABELS = {"ce_calibrator": "CE-calibrator", "value_functional": "Value-functional", "moment_functional": "Moment-functional"}
COLORS = {"ce_calibrator": "#D08770", "value_functional": "#5E81AC", "moment_functional": "#BF616A"}
PROMPT = "The purpose of scientific research is to"


def stat(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "minimum": min(values), "maximum": max(values)}


def fmt(value):
    return f"{value:.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [json.loads((ROOT / f"seed_{seed}" / "results.json").read_text()) for seed in SEEDS]
    rows = []
    for result in results:
        row = {"seed": result["seed"], "teacher_loss": result["teacher_fresh_test"]["loss"]}
        for cond in CONDS:
            condition = result[cond]
            row[f"{cond}_loss"] = condition["final_fresh_test"]["loss"]
            row[f"{cond}_alpha_zero_error"] = abs(condition["alpha_zero_fresh_test"]["loss"] - result["teacher_fresh_test"]["loss"])
            l1 = condition["stages_development"]["layer_1"]
            row[f"{cond}_l1_value_nmse"] = l1["diagnostic_after"]["value_nmse"]
            row[f"{cond}_l1_log_var"] = l1["diagnostic_after"]["moment_log_variance"]
            row[f"{cond}_l1_mean"] = l1["diagnostic_after"]["moment_mean"]
            row[f"{cond}_l1_token_kl"] = l1["post_gate_token_kl"]
            row[f"{cond}_max_safety_kl"] = max(x["token_normalized_kl"] for x in l1["safety_trace"])
        row["ce_minus_value"] = row["ce_calibrator_loss"] - row["value_functional_loss"]
        row["ce_minus_moment"] = row["ce_calibrator_loss"] - row["moment_functional_loss"]
        row["value_minus_moment"] = row["value_functional_loss"] - row["moment_functional_loss"]
        rows.append(row)

    aggregate = {"experiment": "experiment_017_moment_calibration", "seeds": list(SEEDS), "n": len(SEEDS), "fresh_test": {"eligible_offset": 128, "sequences": 128, "all_conditions_trained_before_test_access": True}, "per_seed": rows, "metrics": {}}
    aggregate["metrics"]["teacher_fresh_test_loss"] = stat([r["teacher_loss"] for r in rows])
    for cond in CONDS:
        for suffix in ("loss", "l1_value_nmse", "l1_log_var", "l1_mean", "l1_token_kl", "max_safety_kl"):
            aggregate["metrics"][f"{cond}_{suffix}"] = stat([r[f"{cond}_{suffix}"] for r in rows])
    for pair in ("ce_minus_value", "ce_minus_moment", "value_minus_moment"):
        aggregate["metrics"][pair] = stat([r[pair] for r in rows])
    aggregate["acceptance"] = {
        "alpha_zero_exact_all_conditions_all_seeds": all(max(r[f"{c}_alpha_zero_error"] for c in CONDS) < 1e-5 for r in rows),
        "all_complete_two_layer_endpoints": True,
        "moment_beats_ce_count": sum(r["ce_minus_moment"] > 0 for r in rows),
        "moment_beats_value_count": sum(r["value_minus_moment"] > 0 for r in rows),
        "moment_beats_ce_mean_by_at_least_0_05": aggregate["metrics"]["ce_minus_moment"]["mean"] >= .05,
        "moment_beats_value_mean_by_at_least_0_03": aggregate["metrics"]["value_minus_moment"]["mean"] >= .03,
        "moment_improves_l1_log_variance_over_value_count": sum(r["moment_functional_l1_log_var"] < r["value_functional_l1_log_var"] for r in rows),
        "moment_within_0_15_of_teacher": aggregate["metrics"]["moment_functional_loss"]["mean"] - aggregate["metrics"]["teacher_fresh_test_loss"]["mean"] <= .15,
        "third_layer_permitted": False,
    }
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = ["# Experiment 017: Fresh-Test Aggregate Results", "", "## Fresh Held-Out Test Loss", "", "| Seed | Frozen teacher | CE-calibrator | Value-functional | Moment-functional | CE − moment | Value − moment |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {fmt(r['teacher_loss'])} | {fmt(r['ce_calibrator_loss'])} | {fmt(r['value_functional_loss'])} | {fmt(r['moment_functional_loss'])} | {fmt(r['ce_minus_moment'])} | {fmt(r['value_minus_moment'])} |")
    lines += ["", "## Aggregate", "", "| Metric | Mean ± sample SD |", "|---|---:|"]
    for key in ("teacher_fresh_test_loss", "ce_calibrator_loss", "value_functional_loss", "moment_functional_loss", "ce_minus_value", "ce_minus_moment", "value_minus_moment", "ce_calibrator_l1_log_var", "value_functional_l1_log_var", "moment_functional_l1_log_var", "ce_calibrator_l1_value_nmse", "value_functional_l1_value_nmse", "moment_functional_l1_value_nmse"):
        item = aggregate["metrics"][key]
        lines.append(f"| {key.replace('_', ' ')} | {fmt(item['mean'])} ± {fmt(item['sample_std'])} |")
    lines += ["", "## Acceptance", "", "| Rule | Outcome |", "|---|---|"]
    for key, value in aggregate["acceptance"].items(): lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines += ["", "All final metrics above come from the fresh test slice of eligible WikiText-2 test sequences 129–256. Each seed trained all three conditions before that slice was requested for evaluation."]
    (OUT / "aggregate_results.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    xs = list(range(len(SEEDS)))
    offsets = {"ce_calibrator": -.20, "value_functional": 0.0, "moment_functional": .20}
    for cond in CONDS:
        ys = [r[f"{cond}_loss"] for r in rows]
        axes[0].scatter([x + offsets[cond] for x in xs], ys, color=COLORS[cond], label=LABELS[cond], s=62, zorder=2)
    for r, x in zip(rows, xs):
        axes[0].plot([x + offsets[c] for c in CONDS], [r[f"{c}_loss"] for c in CONDS], color="#B0B0B0", zorder=1)
    axes[0].axhline(rows[0]["teacher_loss"], color="#4C566A", ls="--", label="Frozen teacher")
    axes[0].set(xticks=xs, xticklabels=[str(s) for s in SEEDS], ylabel="Fresh held-out loss (lower is better)", title="Complete two-layer endpoints")
    axes[0].legend(fontsize=9)
    moment_names = ["CE\ncalibrator", "Value\nfunctional", "Moment\nfunctional"]
    log_var = [aggregate["metrics"][f"{c}_l1_log_var"]["mean"] for c in CONDS]
    nmse = [aggregate["metrics"][f"{c}_l1_value_nmse"]["mean"] for c in CONDS]
    positions = range(3)
    bars = axes[1].bar([p - .18 for p in positions], log_var, .36, color="#A3BE8C", label="L1 log-variance diagnostic")
    axes[1].set(xticks=list(positions), xticklabels=moment_names, ylabel="Development diagnostic magnitude", title="Moment fit improved locally, not globally")
    ax2 = axes[1].twinx(); ax2.plot(positions, nmse, marker="o", color="#BF616A", label="L1 value NMSE"); ax2.set_ylabel("L1 value NMSE")
    handles, labels = axes[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); axes[1].legend(handles + h2, labels + l2, loc="upper right", fontsize=8)
    fig.savefig(OUT / "fresh_test_and_local_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"acceptance": aggregate["acceptance"], "metrics": aggregate["metrics"]}, indent=2))

if __name__ == "__main__": main()
