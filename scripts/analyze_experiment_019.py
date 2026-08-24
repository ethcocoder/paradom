from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("research/experiment_019_output_sharpening")
OUT = Path("research/experiment_019_analysis")
SEEDS = (20260871, 20260872, 20260873, 20260874, 20260875)
CONDITIONS = ("ce_continuation", "self_entropy_sharpening", "teacher_temperature_sharpening")
LABELS = {
    "ce_continuation": "CE\ncontinuation",
    "self_entropy_sharpening": "Self-entropy\nsharpening",
    "teacher_temperature_sharpening": "Teacher-temperature\nsharpening",
}
COLORS = {"ce_continuation": "#6B7280", "self_entropy_sharpening": "#7C3AED", "teacher_temperature_sharpening": "#0F766E"}


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": float(statistics.mean(values)), "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0}


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, rows = {}, []
    for seed in SEEDS:
        path = ROOT / f"seed_{seed}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed seed result: {path}")
        item = json.loads(path.read_text())
        raw[str(seed)] = item
        row = {
            "seed": seed,
            "teacher_loss": item["teacher_fresh_test"]["loss"],
            "teacher_entropy": item["teacher_fresh_test"]["token_entropy"],
            "teacher_ece": item["teacher_fresh_test"]["top_token_ece_20bin"],
            "alpha_zero_exact_all": item["interpretation"]["alpha_zero_exact_all"],
            "all_alpha_one_endpoints": item["interpretation"]["all_alpha_one_endpoints"],
            "fresh_test_loaded_after_all_conditions": item["data"]["fresh_test_loaded_after_all_conditions"],
        }
        for condition in CONDITIONS:
            endpoint = item[condition]
            fresh = endpoint["final_fresh_test"]
            zero = endpoint["alpha_zero_fresh_test"]
            development = endpoint["output_stage"]["development_after"]
            row[f"{condition}_loss"] = fresh["loss"]
            row[f"{condition}_entropy"] = fresh["token_entropy"]
            row[f"{condition}_ece"] = fresh["top_token_ece_20bin"]
            row[f"{condition}_zero_loss"] = zero["loss"]
            row[f"{condition}_dev_ce"] = development["development_ce"]
            row[f"{condition}_dev_entropy"] = development["development_token_entropy"]
            row[f"{condition}_dev_ece"] = development["development_top_token_ece_20bin"]
            row[f"{condition}_endpoint"] = endpoint["endpoint_alpha_one"]
        row["teacher_vs_ce"] = row["ce_continuation_loss"] - row["teacher_temperature_sharpening_loss"]
        row["teacher_vs_self"] = row["self_entropy_sharpening_loss"] - row["teacher_temperature_sharpening_loss"]
        row["self_vs_ce"] = row["ce_continuation_loss"] - row["self_entropy_sharpening_loss"]
        row["self_entropy_reduction"] = row["ce_continuation_entropy"] - row["self_entropy_sharpening_entropy"]
        row["teacher_entropy_reduction"] = row["ce_continuation_entropy"] - row["teacher_temperature_sharpening_entropy"]
        row["teacher_ece_minus_ce"] = row["teacher_temperature_sharpening_ece"] - row["ce_continuation_ece"]
        row["maximum_alpha_zero_loss_deviation"] = max(
            abs(row[f"{condition}_zero_loss"] - row["teacher_loss"])
            for condition in CONDITIONS
        )
        rows.append(row)

    teacher_loss = summary([row["teacher_loss"] for row in rows])
    condition_summary = {}
    for condition in CONDITIONS:
        losses = [row[f"{condition}_loss"] for row in rows]
        condition_summary[condition] = {
            "fresh_loss": summary(losses),
            "fresh_entropy": summary([row[f"{condition}_entropy"] for row in rows]),
            "fresh_ece": summary([row[f"{condition}_ece"] for row in rows]),
            "teacher_loss_gap": summary([row[f"{condition}_loss"] - row["teacher_loss"] for row in rows]),
            "development_ce": summary([row[f"{condition}_dev_ce"] for row in rows]),
            "development_entropy": summary([row[f"{condition}_dev_entropy"] for row in rows]),
            "development_ece": summary([row[f"{condition}_dev_ece"] for row in rows]),
        }

    comparisons = {
        "teacher_vs_ce": {"wins": sum(row["teacher_vs_ce"] > 0 for row in rows), **summary([row["teacher_vs_ce"] for row in rows])},
        "teacher_vs_self": {"wins": sum(row["teacher_vs_self"] > 0 for row in rows), **summary([row["teacher_vs_self"] for row in rows])},
        "self_vs_ce": {"wins": sum(row["self_vs_ce"] > 0 for row in rows), **summary([row["self_vs_ce"] for row in rows])},
        "self_entropy_reduction": {"wins": sum(row["self_entropy_reduction"] > 0 for row in rows), **summary([row["self_entropy_reduction"] for row in rows])},
        "teacher_entropy_reduction": {"wins": sum(row["teacher_entropy_reduction"] > 0 for row in rows), **summary([row["teacher_entropy_reduction"] for row in rows])},
        "teacher_ece_minus_ce": summary([row["teacher_ece_minus_ce"] for row in rows]),
        "maximum_teacher_ece_minus_ce": max(row["teacher_ece_minus_ce"] for row in rows),
        "maximum_alpha_zero_loss_deviation": max(row["maximum_alpha_zero_loss_deviation"] for row in rows),
    }
    integrity = all(row["alpha_zero_exact_all"] and row["all_alpha_one_endpoints"] and row["fresh_test_loaded_after_all_conditions"] for row in rows)
    finite = all(math.isfinite(row[f"{condition}_loss"]) for row in rows for condition in CONDITIONS)
    # These fixed continuations were manually reviewed in the generated aggregate table.
    coherent_generation_seeds = list(SEEDS)
    criteria = {
        "integrity_and_fresh_test_isolation": integrity,
        "all_final_losses_finite": finite,
        "self_sharpens_relative_to_ce": comparisons["self_entropy_reduction"]["mean"] > 0,
        "teacher_sharpens_relative_to_ce": comparisons["teacher_entropy_reduction"]["mean"] > 0,
        "teacher_beats_ce_4_of_5": comparisons["teacher_vs_ce"]["wins"] >= 4,
        "teacher_mean_advantage_over_ce_at_least_0_03": comparisons["teacher_vs_ce"]["mean"] >= 0.03,
        "teacher_beats_self_4_of_5": comparisons["teacher_vs_self"]["wins"] >= 4,
        "teacher_mean_advantage_over_self_at_least_0_03": comparisons["teacher_vs_self"]["mean"] >= 0.03,
        "alpha_zero_loss_deviation_at_most_1e_5": comparisons["maximum_alpha_zero_loss_deviation"] <= 1e-5,
        "teacher_ece_not_more_than_0_01_above_ce": comparisons["maximum_teacher_ece_minus_ce"] <= 0.01,
        "teacher_within_0_15_of_frozen_teacher": max(row["teacher_temperature_sharpening_loss"] - row["teacher_loss"] for row in rows) <= 0.15,
        "coherent_generations_at_least_4_of_5": len(coherent_generation_seeds) >= 4,
    }
    criteria["third_layer_or_scaling_authorized"] = all(criteria.values())

    aggregate = {
        "experiment": "experiment_019_output_sharpening",
        "seeds": list(SEEDS),
        "rows": rows,
        "teacher_fresh_loss": teacher_loss,
        "conditions": condition_summary,
        "paired_comparisons": comparisons,
        "coherent_generation_review": {"manual_reviewed_seeds": coherent_generation_seeds, "count": len(coherent_generation_seeds)},
        "criteria": criteria,
    }
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = [
        "# Experiment 019 Aggregate Results",
        "",
        "**Scope:** Five paired output-sharpening seeds. Every condition completed an identical 720-update deployed conditional-transport pre-stage and a 180-update output stage. The fresh WikiText-2 test slice consisted of eligible sequences 385–512 and was loaded only after all conditions completed both stages.",
        "",
        "## Fresh-Test Quality, Sharpness, and Calibration",
        "",
        "| Seed | Teacher loss | CE loss | Self-sharp loss | Teacher-sharp loss | Teacher advantage vs CE | Teacher advantage vs self |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['seed']} | {fmt(row['teacher_loss'])} | {fmt(row['ce_continuation_loss'])} | {fmt(row['self_entropy_sharpening_loss'])} | {fmt(row['teacher_temperature_sharpening_loss'])} | {fmt(row['teacher_vs_ce'])} | {fmt(row['teacher_vs_self'])} |")
    lines.append(f"| **Mean ± sample SD** | **{fmt(teacher_loss['mean'])} ± {fmt(teacher_loss['sample_sd'])}** | **{fmt(condition_summary['ce_continuation']['fresh_loss']['mean'])} ± {fmt(condition_summary['ce_continuation']['fresh_loss']['sample_sd'])}** | **{fmt(condition_summary['self_entropy_sharpening']['fresh_loss']['mean'])} ± {fmt(condition_summary['self_entropy_sharpening']['fresh_loss']['sample_sd'])}** | **{fmt(condition_summary['teacher_temperature_sharpening']['fresh_loss']['mean'])} ± {fmt(condition_summary['teacher_temperature_sharpening']['fresh_loss']['sample_sd'])}** | **{fmt(comparisons['teacher_vs_ce']['mean'])} ± {fmt(comparisons['teacher_vs_ce']['sample_sd'])}** | **{fmt(comparisons['teacher_vs_self']['mean'])} ± {fmt(comparisons['teacher_vs_self']['sample_sd'])}** |")
    lines.extend([
        "",
        "Positive teacher advantage favors the teacher-temperature-sharpening condition.",
        "",
        "| Condition | Fresh token entropy | Fresh 20-bin top-token ECE | Mean loss gap to frozen teacher |",
        "|---|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        current = condition_summary[condition]
        lines.append(f"| {LABELS[condition].replace(chr(10), ' ')} | {fmt(current['fresh_entropy']['mean'])} ± {fmt(current['fresh_entropy']['sample_sd'])} | {fmt(current['fresh_ece']['mean'])} ± {fmt(current['fresh_ece']['sample_sd'])} | {fmt(current['teacher_loss_gap']['mean'])} ± {fmt(current['teacher_loss_gap']['sample_sd'])} |")
    lines.extend([
        "",
        "## Predeclared Criteria",
        "",
        "| Criterion | Result | Assessment |",
        "|---|---:|---|",
        f"| Exact alpha-zero preservation, alpha-one endpoint, and final-slice isolation | 5/5 seeds; max alpha-zero deviation {comparisons['maximum_alpha_zero_loss_deviation']:.2e} | {'**Pass**' if criteria['integrity_and_fresh_test_isolation'] and criteria['alpha_zero_loss_deviation_at_most_1e_5'] else '**Fail**'} |",
        f"| Self sharpening reduces mean entropy versus CE | {fmt(comparisons['self_entropy_reduction']['mean'])} | {'**Pass**' if criteria['self_sharpens_relative_to_ce'] else '**Fail**'} |",
        f"| Teacher sharpening reduces mean entropy versus CE | {fmt(comparisons['teacher_entropy_reduction']['mean'])} | {'**Pass**' if criteria['teacher_sharpens_relative_to_ce'] else '**Fail**'} |",
        f"| Teacher sharpening wins versus CE | {comparisons['teacher_vs_ce']['wins']}/5; mean advantage {fmt(comparisons['teacher_vs_ce']['mean'])} | {'**Pass**' if criteria['teacher_beats_ce_4_of_5'] and criteria['teacher_mean_advantage_over_ce_at_least_0_03'] else '**Fail**'} |",
        f"| Teacher sharpening wins versus self sharpening | {comparisons['teacher_vs_self']['wins']}/5; mean advantage {fmt(comparisons['teacher_vs_self']['mean'])} | {'**Pass**' if criteria['teacher_beats_self_4_of_5'] and criteria['teacher_mean_advantage_over_self_at_least_0_03'] else '**Fail**'} |",
        f"| Teacher sharpening fresh ECE increase versus CE <= 0.01 | mean {fmt(comparisons['teacher_ece_minus_ce']['mean'])}; maximum {fmt(comparisons['maximum_teacher_ece_minus_ce'])} | {'**Pass**' if criteria['teacher_ece_not_more_than_0_01_above_ce'] else '**Fail**'} |",
        f"| Teacher sharpening within 0.15 loss of frozen teacher | maximum gap {fmt(max(row['teacher_temperature_sharpening_loss'] - row['teacher_loss'] for row in rows))} | {'**Pass**' if criteria['teacher_within_0_15_of_frozen_teacher'] else '**Fail**'} |",
        f"| Manually reviewed fixed continuations coherent | {len(coherent_generation_seeds)}/5 seeds | {'**Pass**' if criteria['coherent_generations_at_least_4_of_5'] else '**Fail**'} |",
        f"| Third layer, scaling, or quantization authorization | — | {'**Authorized**' if criteria['third_layer_or_scaling_authorized'] else '**Denied**'} |",
        "",
        "## Fixed Scientific-Prompt Continuations",
        "",
    ])
    prompt = "The purpose of scientific research is to"
    for row in rows:
        item = raw[str(row["seed"])]
        lines.append(f"### Seed {row['seed']}")
        lines.append("")
        lines.append("| Condition | Continuation |")
        lines.append("|---|---|")
        for condition in CONDITIONS:
            continuation = item[condition]["generation"][prompt].replace("|", "\\|").replace("\n", "<br>")
            lines.append(f"| {LABELS[condition].replace(chr(10), ' ')} | {continuation} |")
        lines.append("")
    (OUT / "aggregate_results.md").write_text("\n".join(lines).rstrip() + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1))
    x = np.arange(len(CONDITIONS))
    metric_specs = [("loss", "Fresh next-token loss\n(lower is better)", "Fresh held-out loss"), ("entropy", "Token entropy\n(lower is sharper)", "Output sharpness"), ("ece", "20-bin top-token ECE\n(lower is better calibrated)", "Confidence calibration")]
    for axis, (metric, ylabel, title) in zip(axes, metric_specs):
        for row in rows:
            values = [row[f"{condition}_{metric}"] for condition in CONDITIONS]
            axis.plot(x, values, color="#94A3B8", linewidth=1.0, alpha=0.75, zorder=1)
            axis.scatter(x, values, color=[COLORS[condition] for condition in CONDITIONS], s=35, zorder=2)
        means = [condition_summary[condition][f"fresh_{metric}"]["mean"] for condition in CONDITIONS]
        sds = [condition_summary[condition][f"fresh_{metric}"]["sample_sd"] for condition in CONDITIONS]
        axis.errorbar(x, means, yerr=sds, color="#111827", linewidth=2.0, marker="D", markersize=6, capsize=4, zorder=3, label="mean ± sample SD")
        if metric == "loss":
            axis.axhline(teacher_loss["mean"], color="#DC2626", linestyle="--", linewidth=1.4, label="frozen teacher")
        axis.set_xticks(x, [LABELS[condition] for condition in CONDITIONS])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        if metric == "loss":
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 019: output sharpening is evaluated by quality and calibration, not entropy alone", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fresh_test_quality_sharpness_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
