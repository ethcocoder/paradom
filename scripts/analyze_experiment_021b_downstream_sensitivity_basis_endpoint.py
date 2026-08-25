from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("research/experiment_021b_downstream_sensitivity_basis_endpoint")
OUT = Path("research/experiment_021b_downstream_sensitivity_basis_analysis")
SEEDS = (20260911, 20260912, 20260913, 20260914, 20260915)
CONDITIONS = ("unit_conditional_ce_transport", "correct_sensitivity_basis", "permuted_sensitivity_basis")
LABELS = {"unit_conditional_ce_transport": "Unit CE\ntransport", "correct_sensitivity_basis": "Correct\nsensitivity basis", "permuted_sensitivity_basis": "Permuted\nsensitivity basis"}
COLORS = {"unit_conditional_ce_transport": "#6B7280", "correct_sensitivity_basis": "#0F766E", "permuted_sensitivity_basis": "#A16207"}


def stats(values: list[float]) -> dict[str, float]:
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
            raise FileNotFoundError(f"Missing endpoint result: {path}")
        item = json.loads(path.read_text())
        raw[str(seed)] = item
        row = {"seed": seed, "teacher_loss": item["teacher_fresh_test"]["loss"], "alpha_zero_exact_all": item["interpretation"]["alpha_zero_exact_all"], "all_alpha_one_endpoints": item["interpretation"]["all_alpha_one_endpoints"], "fresh_test_loaded_after_all_conditions": item["data"]["fresh_test_loaded_after_all_conditions"]}
        for condition in CONDITIONS:
            current, fresh, dev = item[condition], item[condition]["final_fresh_test"], item[condition]["development_endpoint"]
            row[f"{condition}_loss"] = fresh["loss"]
            row[f"{condition}_entropy"] = fresh["token_entropy"]
            row[f"{condition}_ece"] = fresh["top_token_ece_20bin"]
            row[f"{condition}_alpha_zero"] = current["alpha_zero_loss_deviation"]
            row[f"{condition}_endpoint"] = current["endpoint_alpha_one"]
            row[f"{condition}_finite"] = finite_tree(fresh)
            row[f"{condition}_dev_ce"] = dev["development_ce"]
            row[f"{condition}_weighted"] = dev["sensitivity_weighted_attention_output_discrepancy"]
            row[f"{condition}_unweighted"] = dev["unweighted_attention_output_mse"]
            row[f"{condition}_drift_l1"] = dev["post_layer_1_relative_drift"]
            row[f"{condition}_drift_l2"] = dev["post_layer_2_relative_drift"]
        row["correct_vs_unit"] = row["unit_conditional_ce_transport_loss"] - row["correct_sensitivity_basis_loss"]
        row["correct_vs_permuted"] = row["permuted_sensitivity_basis_loss"] - row["correct_sensitivity_basis_loss"]
        row["correct_ece_minus_unit"] = row["correct_sensitivity_basis_ece"] - row["unit_conditional_ce_transport_ece"]
        rows.append(row)

    condition_summary = {condition: {"fresh_loss": stats([row[f"{condition}_loss"] for row in rows]), "fresh_entropy": stats([row[f"{condition}_entropy"] for row in rows]), "fresh_ece": stats([row[f"{condition}_ece"] for row in rows]), "teacher_gap": stats([row[f"{condition}_loss"] - row["teacher_loss"] for row in rows]), "development_ce": stats([row[f"{condition}_dev_ce"] for row in rows]), "weighted_discrepancy": stats([row[f"{condition}_weighted"] for row in rows]), "unweighted_mse": stats([row[f"{condition}_unweighted"] for row in rows]), "post_layer_1_drift": stats([row[f"{condition}_drift_l1"] for row in rows]), "post_layer_2_drift": stats([row[f"{condition}_drift_l2"] for row in rows])} for condition in CONDITIONS}
    comparisons = {"correct_vs_unit": {"wins": sum(row["correct_vs_unit"] > 0 for row in rows), **stats([row["correct_vs_unit"] for row in rows])}, "correct_vs_permuted": {"wins": sum(row["correct_vs_permuted"] > 0 for row in rows), **stats([row["correct_vs_permuted"] for row in rows])}, "correct_ece_minus_unit": stats([row["correct_ece_minus_unit"] for row in rows]), "maximum_correct_ece_minus_unit": max(row["correct_ece_minus_unit"] for row in rows)}
    max_alpha_zero = max(row[f"{condition}_alpha_zero"] for row in rows for condition in CONDITIONS)
    criteria = {"integrity_and_final_slice_isolation": all(row["alpha_zero_exact_all"] and row["all_alpha_one_endpoints"] and row["fresh_test_loaded_after_all_conditions"] for row in rows), "alpha_zero_deviation_at_most_1e_5": max_alpha_zero <= 1e-5, "all_fresh_losses_finite": all(row[f"{condition}_finite"] for row in rows for condition in CONDITIONS), "correct_beats_unit_4_of_5": comparisons["correct_vs_unit"]["wins"] >= 4, "correct_mean_advantage_over_unit_at_least_0_03": comparisons["correct_vs_unit"]["mean"] >= 0.03, "correct_beats_permuted_4_of_5": comparisons["correct_vs_permuted"]["wins"] >= 4, "correct_mean_advantage_over_permuted_at_least_0_03": comparisons["correct_vs_permuted"]["mean"] >= 0.03, "correct_ece_not_more_than_0_01_above_unit": comparisons["maximum_correct_ece_minus_unit"] <= 0.01, "correct_within_0_15_of_teacher": max(row["correct_sensitivity_basis_loss"] - row["teacher_loss"] for row in rows) <= 0.15}
    criteria["third_layer_or_scaling_authorized"] = all(criteria.values())
    aggregate = {"experiment": "experiment_021b_downstream_sensitivity_basis_endpoint", "seeds": list(SEEDS), "rows": rows, "conditions": condition_summary, "paired_comparisons": comparisons, "maximum_alpha_zero_loss_deviation": max_alpha_zero, "criteria": criteria}
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = ["# Experiment 021B Aggregate Results", "", "**Scope:** Five paired endpoint seeds. The new WikiText-2 fresh final range is eligible test sequences 641–768, loaded only after all three conditions trained in every seed.", "", "## Fresh Primary Endpoint", "", "Positive advantages favor correctly ordered downstream-sensitivity basis B.", "", "| Seed | Teacher loss | A: Unit CE | B: Correct basis | C: Permuted basis | A − B | C − B |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {fmt(row['teacher_loss'])} | {fmt(row['unit_conditional_ce_transport_loss'])} | {fmt(row['correct_sensitivity_basis_loss'])} | {fmt(row['permuted_sensitivity_basis_loss'])} | {fmt(row['correct_vs_unit'])} | {fmt(row['correct_vs_permuted'])} |")
    teacher_stat = stats([row["teacher_loss"] for row in rows])
    lines.append(f"| **Mean ± sample SD** | **{fmt(teacher_stat['mean'])} ± {fmt(teacher_stat['sample_sd'])}** | **{fmt(condition_summary['unit_conditional_ce_transport']['fresh_loss']['mean'])} ± {fmt(condition_summary['unit_conditional_ce_transport']['fresh_loss']['sample_sd'])}** | **{fmt(condition_summary['correct_sensitivity_basis']['fresh_loss']['mean'])} ± {fmt(condition_summary['correct_sensitivity_basis']['fresh_loss']['sample_sd'])}** | **{fmt(condition_summary['permuted_sensitivity_basis']['fresh_loss']['mean'])} ± {fmt(condition_summary['permuted_sensitivity_basis']['fresh_loss']['sample_sd'])}** | **{fmt(comparisons['correct_vs_unit']['mean'])} ± {fmt(comparisons['correct_vs_unit']['sample_sd'])}** | **{fmt(comparisons['correct_vs_permuted']['mean'])} ± {fmt(comparisons['correct_vs_permuted']['sample_sd'])}** |")
    lines.extend(["", "## Acceptance Checks", "", "| Criterion | Result | Assessment |", "|---|---:|---|"])
    checks = [("Alpha-zero integrity, alpha-one completion, and final-slice isolation", f"5/5; maximum deviation {max_alpha_zero:.2e}", criteria["integrity_and_final_slice_isolation"] and criteria["alpha_zero_deviation_at_most_1e_5"]), ("Correct basis wins versus unit", f"{comparisons['correct_vs_unit']['wins']}/5; mean advantage {fmt(comparisons['correct_vs_unit']['mean'])}", criteria["correct_beats_unit_4_of_5"] and criteria["correct_mean_advantage_over_unit_at_least_0_03"]), ("Correct basis wins versus permuted", f"{comparisons['correct_vs_permuted']['wins']}/5; mean advantage {fmt(comparisons['correct_vs_permuted']['mean'])}", criteria["correct_beats_permuted_4_of_5"] and criteria["correct_mean_advantage_over_permuted_at_least_0_03"]), ("Correct-basis ECE increase versus unit <= 0.01", f"maximum {fmt(comparisons['maximum_correct_ece_minus_unit'])}", criteria["correct_ece_not_more_than_0_01_above_unit"]), ("Correct basis within 0.15 loss of teacher", f"maximum gap {fmt(max(row['correct_sensitivity_basis_loss'] - row['teacher_loss'] for row in rows))}", criteria["correct_within_0_15_of_teacher"]), ("Third layer / scaling / quantization", "All prior criteria required", criteria["third_layer_or_scaling_authorized"])]
    for name, outcome, passed in checks:
        lines.append(f"| {name} | {outcome} | {'**Pass**' if passed else '**Fail**' if name != 'Third layer / scaling / quantization' else '**Authorized**' if passed else '**Denied**'} |")
    lines.extend(["", "## Development Mechanism Diagnostics", "", "| Condition | Development CE | Weighted discrepancy | Unweighted MSE | Post-layer-2 drift |", "|---|---:|---:|---:|---:|"])
    for condition in CONDITIONS:
        item = condition_summary[condition]
        lines.append(f"| {LABELS[condition].replace(chr(10), ' ')} | {fmt(item['development_ce']['mean'])} ± {fmt(item['development_ce']['sample_sd'])} | {fmt(item['weighted_discrepancy']['mean'])} ± {fmt(item['weighted_discrepancy']['sample_sd'])} | {fmt(item['unweighted_mse']['mean'])} ± {fmt(item['unweighted_mse']['sample_sd'])} | {fmt(item['post_layer_2_drift']['mean'])} ± {fmt(item['post_layer_2_drift']['sample_sd'])} |")
    lines.extend(["", "## Fixed Scientific-Prompt Continuations", ""])
    prompt = "The purpose of scientific research is to"
    for row in rows:
        lines.extend([f"### Seed {row['seed']}", "", "| Condition | Continuation |", "|---|---|"])
        for condition in CONDITIONS:
            continuation = raw[str(row["seed"])][condition]["generation"][prompt].replace("|", "\\|").replace("\n", "<br>")
            lines.append(f"| {LABELS[condition].replace(chr(10), ' ')} | {continuation} |")
        lines.append("")
    (OUT / "aggregate_results.md").write_text("\n".join(lines).rstrip() + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1))
    x = np.arange(len(CONDITIONS))
    teacher_mean = teacher_stat["mean"]
    for axis, metric, ylabel, title in zip(axes, ("loss", "entropy", "ece"), ("Fresh next-token loss\n(lower is better)", "Fresh token entropy\n(lower is sharper)", "Fresh 20-bin top-token ECE\n(lower is better calibrated)"), ("Fresh held-out language loss", "Output sharpness diagnostic", "Calibration diagnostic")):
        for row in rows:
            values = [row[f"{condition}_{metric}"] for condition in CONDITIONS]
            axis.plot(x, values, color="#94A3B8", linewidth=1.0, alpha=0.75, zorder=1)
            axis.scatter(x, values, color=[COLORS[condition] for condition in CONDITIONS], s=35, zorder=2)
        means = [condition_summary[condition][f"fresh_{metric}"]["mean"] for condition in CONDITIONS]
        sds = [condition_summary[condition][f"fresh_{metric}"]["sample_sd"] for condition in CONDITIONS]
        axis.errorbar(x, means, yerr=sds, color="#111827", linewidth=2, marker="D", capsize=4, zorder=3, label="mean ± sample SD")
        if metric == "loss":
            axis.axhline(teacher_mean, color="#DC2626", linestyle="--", linewidth=1.4, label="frozen teacher")
            axis.legend(frameon=False, fontsize=8)
        axis.set_xticks(x, [LABELS[condition] for condition in CONDITIONS]); axis.set_ylabel(ylabel); axis.set_title(title); axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Experiment 021B: correct downstream sensitivity must beat unit and topology-destroyed bases", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fresh_test_sensitivity_basis_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
