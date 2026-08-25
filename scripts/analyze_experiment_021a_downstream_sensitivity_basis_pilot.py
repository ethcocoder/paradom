from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("research/experiment_021a_downstream_sensitivity_basis_pilot")
OUT = Path("research/experiment_021a_downstream_sensitivity_basis_analysis")
SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
CONDITIONS = ("unit_conditional_ce_transport", "correct_sensitivity_basis", "permuted_sensitivity_basis")
LABELS = {"unit_conditional_ce_transport": "Unit CE\ntransport", "correct_sensitivity_basis": "Correct\nsensitivity basis", "permuted_sensitivity_basis": "Permuted\nsensitivity basis"}
COLORS = {"unit_conditional_ce_transport": "#6B7280", "correct_sensitivity_basis": "#0F766E", "permuted_sensitivity_basis": "#A16207"}


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": float(statistics.mean(values)), "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0}


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (int, bool, str)) or value is None:
        return True
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def fmt(value: float) -> str:
    return f"{value:.5f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, rows = {}, []
    for seed in SEEDS:
        path = ROOT / f"seed_{seed}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing result: {path}")
        item = json.loads(path.read_text())
        if item["data"]["test_split_requested"] is not False:
            raise RuntimeError(f"Seed {seed} violated the test prohibition")
        raw[str(seed)] = item
        row = {"seed": seed, "test_split_requested": item["data"]["test_split_requested"]}
        for condition in CONDITIONS:
            output = item[condition]
            diagnostic = output["development_endpoint"]
            row[f"{condition}_ce"] = diagnostic["development_ce"]
            row[f"{condition}_weighted"] = diagnostic["sensitivity_weighted_attention_output_discrepancy"]
            row[f"{condition}_unweighted"] = diagnostic["unweighted_attention_output_mse"]
            row[f"{condition}_drift_l1"] = diagnostic["post_layer_1_relative_drift"]
            row[f"{condition}_drift_l2"] = diagnostic["post_layer_2_relative_drift"]
            row[f"{condition}_alpha_zero"] = output["alpha_zero_integrity"]["absolute_loss_deviation"]
            row[f"{condition}_endpoint"] = output["endpoint_alpha_one"]
            row[f"{condition}_finite"] = finite_tree(diagnostic)
        row["correct_ce_minus_unit"] = row["correct_sensitivity_basis_ce"] - row["unit_conditional_ce_transport_ce"]
        row["correct_weighted_minus_unit"] = row["correct_sensitivity_basis_weighted"] - row["unit_conditional_ce_transport_weighted"]
        row["correct_weighted_minus_permuted"] = row["correct_sensitivity_basis_weighted"] - row["permuted_sensitivity_basis_weighted"]
        rows.append(row)

    summaries = {condition: {"development_ce": stats([row[f"{condition}_ce"] for row in rows]), "weighted_discrepancy": stats([row[f"{condition}_weighted"] for row in rows]), "unweighted_mse": stats([row[f"{condition}_unweighted"] for row in rows]), "post_layer_1_drift": stats([row[f"{condition}_drift_l1"] for row in rows]), "post_layer_2_drift": stats([row[f"{condition}_drift_l2"] for row in rows])} for condition in CONDITIONS}
    comparisons = {"correct_ce_minus_unit": stats([row["correct_ce_minus_unit"] for row in rows]), "correct_weighted_minus_unit": stats([row["correct_weighted_minus_unit"] for row in rows]), "correct_weighted_minus_permuted": stats([row["correct_weighted_minus_permuted"] for row in rows])}
    max_alpha_zero = max(row[f"{condition}_alpha_zero"] for row in rows for condition in CONDITIONS)
    criteria = {"no_test_access": all(row["test_split_requested"] is False for row in rows), "alpha_zero_integrity": max_alpha_zero <= 1e-5, "all_endpoints_alpha_one": all(row[f"{condition}_endpoint"] for row in rows for condition in CONDITIONS), "all_endpoint_metrics_finite": all(row[f"{condition}_finite"] for row in rows for condition in CONDITIONS), "correct_ce_safe": comparisons["correct_ce_minus_unit"]["mean"] <= 0.02, "correct_weighted_discrepancy_lower_than_unit": comparisons["correct_weighted_minus_unit"]["mean"] < 0.0, "correct_weighted_discrepancy_lower_than_permuted": comparisons["correct_weighted_minus_permuted"]["mean"] < 0.0}
    criteria["endpoint_protocol_permitted"] = all(criteria.values())
    aggregate = {"experiment": "experiment_021a_downstream_sensitivity_basis_pilot", "scope": "development_only_no_test_access", "seeds": list(SEEDS), "rows": rows, "condition_summaries": summaries, "paired_differences": comparisons, "maximum_alpha_zero_loss_deviation": max_alpha_zero, "criteria": criteria}
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = ["# Experiment 021A Development-Only Aggregate", "", "**Scope:** Five-seed sensitivity-basis feasibility pilot. The test split was prohibited and `test_split_requested` is false in every result. These are development diagnostics, not endpoint transfer evidence.", "", "## Per-Seed Differences", "", "Positive CE difference means worse than the unit-basis CE control. Negative discrepancy difference means lower sensitivity-weighted attention-output discrepancy.", "", "| Seed | Correct CE − unit | Correct weighted − unit | Correct weighted − permuted |", "|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {fmt(row['correct_ce_minus_unit'])} | {fmt(row['correct_weighted_minus_unit'])} | {fmt(row['correct_weighted_minus_permuted'])} |")
    lines.extend(["", "## Locked Feasibility Rule", "", "| Requirement | Result | Assessment |", "|---|---:|---|"])
    entries = [("No test access", "5/5 test flags false", criteria["no_test_access"]), ("Alpha-zero integrity", f"maximum deviation {max_alpha_zero:.2e}", criteria["alpha_zero_integrity"]), ("Finite alpha-one endpoints", "15/15 endpoints", criteria["all_endpoints_alpha_one"] and criteria["all_endpoint_metrics_finite"]), ("Correct basis CE safety", f"mean B−A = {fmt(comparisons['correct_ce_minus_unit']['mean'])}", criteria["correct_ce_safe"]), ("Correct basis lower weighted discrepancy than unit", f"mean B−A = {fmt(comparisons['correct_weighted_minus_unit']['mean'])}", criteria["correct_weighted_discrepancy_lower_than_unit"]), ("Correct basis lower weighted discrepancy than permuted", f"mean B−C = {fmt(comparisons['correct_weighted_minus_permuted']['mean'])}", criteria["correct_weighted_discrepancy_lower_than_permuted"])]
    for name, value, passed in entries:
        lines.append(f"| {name} | {value} | {'**Pass**' if passed else '**Fail**'} |")
    conclusion = "**A separately preregistered endpoint protocol is permitted.** This does not establish transfer success and cannot alter the fixed basis formula." if criteria["endpoint_protocol_permitted"] else "**Endpoint protocol is not permitted.** The mechanism stops without any test access."
    lines.extend(["", conclusion, "", "## Development Diagnostics", "", "| Condition | Development CE | Weighted discrepancy | Unweighted MSE | Post-layer-2 drift |", "|---|---:|---:|---:|---:|"])
    for condition in CONDITIONS:
        item = summaries[condition]
        lines.append(f"| {LABELS[condition].replace(chr(10), ' ')} | {fmt(item['development_ce']['mean'])} ± {fmt(item['development_ce']['sample_sd'])} | {fmt(item['weighted_discrepancy']['mean'])} ± {fmt(item['weighted_discrepancy']['sample_sd'])} | {fmt(item['unweighted_mse']['mean'])} ± {fmt(item['unweighted_mse']['sample_sd'])} | {fmt(item['post_layer_2_drift']['mean'])} ± {fmt(item['post_layer_2_drift']['sample_sd'])} |")
    (OUT / "aggregate_results.md").write_text("\n".join(lines).rstrip() + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(CONDITIONS))
    for ax, metric, ylabel, title in [(axes[0], "ce", "Development next-token CE\n(lower is better)", "CE safety diagnostic"), (axes[1], "weighted", "Sensitivity-weighted output discrepancy\n(lower is better)", "Frozen downstream-sensitivity diagnostic")]:
        for row in rows:
            values = [row[f"{condition}_{metric}"] for condition in CONDITIONS]
            ax.plot(x, values, color="#94A3B8", linewidth=1, alpha=0.75, zorder=1)
            ax.scatter(x, values, color=[COLORS[c] for c in CONDITIONS], s=36, zorder=2)
        means = [summaries[c]["development_ce" if metric == "ce" else "weighted_discrepancy"]["mean"] for c in CONDITIONS]
        sds = [summaries[c]["development_ce" if metric == "ce" else "weighted_discrepancy"]["sample_sd"] for c in CONDITIONS]
        ax.errorbar(x, means, yerr=sds, color="#111827", marker="D", linewidth=2, capsize=4, zorder=3, label="mean ± sample SD")
        ax.set_xticks(x, [LABELS[c] for c in CONDITIONS]); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 021A: development-only downstream-sensitivity basis feasibility", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity_basis_pilot_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
