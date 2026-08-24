from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("research/experiment_020b_relational_geometry_endpoint")
OUT = Path("research/experiment_020b_relational_geometry_analysis")
SEEDS = (20260891, 20260892, 20260893, 20260894, 20260895)
CONDITIONS = ("conditional_ce_transport", "correct_relational_geometry", "permuted_relational_control")
LABELS = {"conditional_ce_transport": "CE\ntransport", "correct_relational_geometry": "Correct\nrelation", "permuted_relational_control": "Permuted\nrelation"}
COLORS = {"conditional_ce_transport": "#6B7280", "correct_relational_geometry": "#0F766E", "permuted_relational_control": "#A16207"}


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": float(statistics.mean(values)), "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0}


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return True
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, rows = {}, []
    for seed in SEEDS:
        path = ROOT / f"seed_{seed}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed endpoint result: {path}")
        item = json.loads(path.read_text())
        raw[str(seed)] = item
        row = {
            "seed": seed,
            "teacher_loss": item["teacher_fresh_test"]["loss"],
            "alpha_zero_exact_all": item["interpretation"]["alpha_zero_exact_all"],
            "all_alpha_one_endpoints": item["interpretation"]["all_alpha_one_endpoints"],
            "fresh_test_loaded_after_all_conditions": item["data"]["fresh_test_loaded_after_all_conditions"],
        }
        for condition in CONDITIONS:
            current = item[condition]
            fresh = current["final_fresh_test"]
            diagnostic = current["development_endpoint"]
            row[f"{condition}_loss"] = fresh["loss"]
            row[f"{condition}_entropy"] = fresh["token_entropy"]
            row[f"{condition}_ece"] = fresh["top_token_ece_20bin"]
            row[f"{condition}_alpha_zero_deviation"] = current["alpha_zero_loss_deviation"]
            row[f"{condition}_endpoint"] = current["endpoint_alpha_one"]
            row[f"{condition}_finite"] = finite_tree(fresh)
            row[f"{condition}_dev_ce"] = diagnostic["development_ce"]
            row[f"{condition}_dev_correct_relation"] = diagnostic["development_correct_relation_kl"]
            row[f"{condition}_dev_permuted_relation"] = diagnostic["development_permuted_relation_kl"]
            row[f"{condition}_drift_l1"] = diagnostic["post_layer_1_relative_drift"]
            row[f"{condition}_drift_l2"] = diagnostic["post_layer_2_relative_drift"]
        row["correct_vs_ce"] = row["conditional_ce_transport_loss"] - row["correct_relational_geometry_loss"]
        row["correct_vs_permuted"] = row["permuted_relational_control_loss"] - row["correct_relational_geometry_loss"]
        row["permuted_vs_ce"] = row["conditional_ce_transport_loss"] - row["permuted_relational_control_loss"]
        row["correct_ece_minus_ce"] = row["correct_relational_geometry_ece"] - row["conditional_ce_transport_ece"]
        rows.append(row)

    condition_summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_summary[condition] = {
            "fresh_loss": summary([row[f"{condition}_loss"] for row in rows]),
            "fresh_entropy": summary([row[f"{condition}_entropy"] for row in rows]),
            "fresh_ece": summary([row[f"{condition}_ece"] for row in rows]),
            "teacher_loss_gap": summary([row[f"{condition}_loss"] - row["teacher_loss"] for row in rows]),
            "development_ce": summary([row[f"{condition}_dev_ce"] for row in rows]),
            "development_correct_relation_kl": summary([row[f"{condition}_dev_correct_relation"] for row in rows]),
            "development_permuted_relation_kl": summary([row[f"{condition}_dev_permuted_relation"] for row in rows]),
            "post_layer_1_drift": summary([row[f"{condition}_drift_l1"] for row in rows]),
            "post_layer_2_drift": summary([row[f"{condition}_drift_l2"] for row in rows]),
        }
    comparisons = {
        "correct_vs_ce": {"wins": sum(row["correct_vs_ce"] > 0 for row in rows), **summary([row["correct_vs_ce"] for row in rows])},
        "correct_vs_permuted": {"wins": sum(row["correct_vs_permuted"] > 0 for row in rows), **summary([row["correct_vs_permuted"] for row in rows])},
        "permuted_vs_ce": {"wins": sum(row["permuted_vs_ce"] > 0 for row in rows), **summary([row["permuted_vs_ce"] for row in rows])},
        "correct_ece_minus_ce": summary([row["correct_ece_minus_ce"] for row in rows]),
        "maximum_correct_ece_minus_ce": max(row["correct_ece_minus_ce"] for row in rows),
    }
    max_alpha_zero = max(row[f"{condition}_alpha_zero_deviation"] for row in rows for condition in CONDITIONS)
    integrity = all(row["alpha_zero_exact_all"] and row["all_alpha_one_endpoints"] and row["fresh_test_loaded_after_all_conditions"] for row in rows)
    criteria = {
        "integrity_and_fresh_test_isolation": integrity,
        "alpha_zero_deviation_at_most_1e_5": max_alpha_zero <= 1e-5,
        "all_fresh_losses_finite": all(row[f"{condition}_finite"] for row in rows for condition in CONDITIONS),
        "correct_beats_ce_4_of_5": comparisons["correct_vs_ce"]["wins"] >= 4,
        "correct_mean_advantage_over_ce_at_least_0_03": comparisons["correct_vs_ce"]["mean"] >= 0.03,
        "correct_beats_permuted_4_of_5": comparisons["correct_vs_permuted"]["wins"] >= 4,
        "correct_mean_advantage_over_permuted_at_least_0_03": comparisons["correct_vs_permuted"]["mean"] >= 0.03,
        "correct_ece_not_more_than_0_01_above_ce": comparisons["maximum_correct_ece_minus_ce"] <= 0.01,
        "correct_within_0_15_of_teacher": max(row["correct_relational_geometry_loss"] - row["teacher_loss"] for row in rows) <= 0.15,
    }
    criteria["third_layer_or_scaling_authorized"] = all(criteria.values())
    aggregate = {
        "experiment": "experiment_020b_relational_geometry_endpoint",
        "seeds": list(SEEDS),
        "rows": rows,
        "conditions": condition_summary,
        "paired_comparisons": comparisons,
        "maximum_alpha_zero_loss_deviation": max_alpha_zero,
        "criteria": criteria,
    }
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = [
        "# Experiment 020B Aggregate Results",
        "",
        "**Scope:** Five paired endpoint seeds. The fresh WikiText-2 test slice contains eligible sequences 513–640 and was requested only after all three conditions completed their 720 updates in every seed.",
        "",
        "## Fresh Primary Endpoint",
        "",
        "Positive correct-relational advantages favor the correctly indexed relational-geometry condition.",
        "",
        "| Seed | Teacher loss | CE loss | Correct relation loss | Permuted relation loss | Correct vs CE | Correct vs permuted |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['seed']} | {fmt(row['teacher_loss'])} | {fmt(row['conditional_ce_transport_loss'])} | {fmt(row['correct_relational_geometry_loss'])} | {fmt(row['permuted_relational_control_loss'])} | {fmt(row['correct_vs_ce'])} | {fmt(row['correct_vs_permuted'])} |")
    lines.append(f"| **Mean ± sample SD** | **{fmt(summary([row['teacher_loss'] for row in rows])['mean'])} ± {fmt(summary([row['teacher_loss'] for row in rows])['sample_sd'])}** | **{fmt(condition_summary['conditional_ce_transport']['fresh_loss']['mean'])} ± {fmt(condition_summary['conditional_ce_transport']['fresh_loss']['sample_sd'])}** | **{fmt(condition_summary['correct_relational_geometry']['fresh_loss']['mean'])} ± {fmt(condition_summary['correct_relational_geometry']['fresh_loss']['sample_sd'])}** | **{fmt(condition_summary['permuted_relational_control']['fresh_loss']['mean'])} ± {fmt(condition_summary['permuted_relational_control']['fresh_loss']['sample_sd'])}** | **{fmt(comparisons['correct_vs_ce']['mean'])} ± {fmt(comparisons['correct_vs_ce']['sample_sd'])}** | **{fmt(comparisons['correct_vs_permuted']['mean'])} ± {fmt(comparisons['correct_vs_permuted']['sample_sd'])}** |")
    lines.extend(["", "## Diagnostics and Acceptance Checks", "", "| Criterion | Result | Assessment |", "|---|---:|---|"])
    lines.extend([
        f"| Alpha-zero exactness, alpha-one endpoints, and final-slice isolation | 5/5; maximum deviation {max_alpha_zero:.2e} | {'**Pass**' if criteria['integrity_and_fresh_test_isolation'] and criteria['alpha_zero_deviation_at_most_1e_5'] else '**Fail**'} |",
        f"| Correct relation wins versus CE | {comparisons['correct_vs_ce']['wins']}/5; mean advantage {fmt(comparisons['correct_vs_ce']['mean'])} | {'**Pass**' if criteria['correct_beats_ce_4_of_5'] and criteria['correct_mean_advantage_over_ce_at_least_0_03'] else '**Fail**'} |",
        f"| Correct relation wins versus permuted relation | {comparisons['correct_vs_permuted']['wins']}/5; mean advantage {fmt(comparisons['correct_vs_permuted']['mean'])} | {'**Pass**' if criteria['correct_beats_permuted_4_of_5'] and criteria['correct_mean_advantage_over_permuted_at_least_0_03'] else '**Fail**'} |",
        f"| Correct relation ECE increase versus CE <= 0.01 | maximum {fmt(comparisons['maximum_correct_ece_minus_ce'])} | {'**Pass**' if criteria['correct_ece_not_more_than_0_01_above_ce'] else '**Fail**'} |",
        f"| Correct relation within 0.15 loss of teacher | maximum gap {fmt(max(row['correct_relational_geometry_loss'] - row['teacher_loss'] for row in rows))} | {'**Pass**' if criteria['correct_within_0_15_of_teacher'] else '**Fail**'} |",
        f"| Third layer / scaling / quantization authorization | — | {'**Authorized**' if criteria['third_layer_or_scaling_authorized'] else '**Denied**'} |",
        "",
        "## Development-Only Mechanism Diagnostics",
        "",
        "| Condition | Development correct-relation KL | Development CE | Post-layer-1 drift | Post-layer-2 drift |",
        "|---|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        item = condition_summary[condition]
        lines.append(f"| {LABELS[condition].replace(chr(10), ' ')} | {fmt(item['development_correct_relation_kl']['mean'])} ± {fmt(item['development_correct_relation_kl']['sample_sd'])} | {fmt(item['development_ce']['mean'])} ± {fmt(item['development_ce']['sample_sd'])} | {fmt(item['post_layer_1_drift']['mean'])} ± {fmt(item['post_layer_1_drift']['sample_sd'])} | {fmt(item['post_layer_2_drift']['mean'])} ± {fmt(item['post_layer_2_drift']['sample_sd'])} |")
    lines.extend(["", "## Fixed Scientific-Prompt Continuations", ""])
    prompt = "The purpose of scientific research is to"
    for row in rows:
        item = raw[str(row["seed"])]
        lines.extend([f"### Seed {row['seed']}", "", "| Condition | Continuation |", "|---|---|"])
        for condition in CONDITIONS:
            continuation = item[condition]["generation"][prompt].replace("|", "\\|").replace("\n", "<br>")
            lines.append(f"| {LABELS[condition].replace(chr(10), ' ')} | {continuation} |")
        lines.append("")
    (OUT / "aggregate_results.md").write_text("\n".join(lines).rstrip() + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1))
    specs = [("loss", "Fresh next-token loss\n(lower is better)", "Fresh held-out language loss"), ("entropy", "Fresh token entropy\n(lower is sharper)", "Output sharpness diagnostic"), ("ece", "Fresh 20-bin top-token ECE\n(lower is better calibrated)", "Calibration diagnostic")]
    x = np.arange(len(CONDITIONS))
    teacher_mean = summary([row["teacher_loss"] for row in rows])["mean"]
    for axis, (metric, ylabel, title) in zip(axes, specs):
        for row in rows:
            values = [row[f"{condition}_{metric}"] for condition in CONDITIONS]
            axis.plot(x, values, color="#94A3B8", linewidth=1.0, alpha=0.75, zorder=1)
            axis.scatter(x, values, color=[COLORS[condition] for condition in CONDITIONS], s=35, zorder=2)
        means = [condition_summary[condition][f"fresh_{metric}"]["mean"] for condition in CONDITIONS]
        sds = [condition_summary[condition][f"fresh_{metric}"]["sample_sd"] for condition in CONDITIONS]
        axis.errorbar(x, means, yerr=sds, color="#111827", linewidth=2.0, marker="D", markersize=6, capsize=4, zorder=3, label="mean ± sample SD")
        if metric == "loss":
            axis.axhline(teacher_mean, color="#DC2626", linestyle="--", linewidth=1.4, label="frozen teacher")
        axis.set_xticks(x, [LABELS[condition] for condition in CONDITIONS])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        if metric == "loss":
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 020B: correct relation topology must beat both CE and topology-destroyed control", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fresh_test_topology_control_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
