from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("research/experiment_018_residual_interface_transport")
OUT = Path("research/experiment_018_analysis")
SEEDS = (20260861, 20260862, 20260863, 20260864, 20260865)
CONDITIONS = (
    "unconditional_ce_transport",
    "conditional_ce_transport",
    "conditional_downstream_anchor",
)
LABELS = {
    "unconditional_ce_transport": "Unconditional\nCE transport",
    "conditional_ce_transport": "Conditional\nCE transport",
    "conditional_downstream_anchor": "Conditional\nanchor transport",
}
COLORS = {
    "unconditional_ce_transport": "#6B7280",
    "conditional_ce_transport": "#2563EB",
    "conditional_downstream_anchor": "#059669",
}


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": float(statistics.mean(values)), "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0}


def finite(value: float) -> bool:
    return math.isfinite(value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    raw = {}
    for seed in SEEDS:
        path = ROOT / f"seed_{seed}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed seed result: {path}")
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
            endpoint = item[condition]
            row[f"{condition}_loss"] = endpoint["final_fresh_test"]["loss"]
            row[f"{condition}_alpha_zero_loss"] = endpoint["alpha_zero_fresh_test"]["loss"]
            layer1 = endpoint["stages_development"]["layer_1"]["development_after"]
            row[f"{condition}_dev_ce"] = layer1["development_ce"]
            row[f"{condition}_dev_kl"] = layer1["development_token_normalized_kl"]
            row[f"{condition}_l1_correction_ratio"] = layer1["transport_correction_rms_ratio"]
            row[f"{condition}_l1_anchor"] = layer1["downstream_anchor"]
            row[f"{condition}_endpoint_alpha_one"] = endpoint["endpoint_alpha_one"]
        row["conditional_advantage"] = row["unconditional_ce_transport_loss"] - row["conditional_ce_transport_loss"]
        row["anchor_advantage"] = row["conditional_ce_transport_loss"] - row["conditional_downstream_anchor_loss"]
        row["anchor_vs_unconditional"] = row["unconditional_ce_transport_loss"] - row["conditional_downstream_anchor_loss"]
        rows.append(row)

    teacher_values = [row["teacher_loss"] for row in rows]
    summaries = {"teacher": mean_sd(teacher_values)}
    for condition in CONDITIONS:
        losses = [row[f"{condition}_loss"] for row in rows]
        summaries[condition] = {
            "fresh_test_loss": mean_sd(losses),
            "teacher_gap": mean_sd([loss - row["teacher_loss"] for loss, row in zip(losses, rows)]),
            "development_l1_ce": mean_sd([row[f"{condition}_dev_ce"] for row in rows]),
            "development_l1_token_kl": mean_sd([row[f"{condition}_dev_kl"] for row in rows]),
            "development_l1_correction_ratio": mean_sd([row[f"{condition}_l1_correction_ratio"] for row in rows]),
            "development_l1_anchor": None if condition != "conditional_downstream_anchor" else mean_sd([row[f"{condition}_l1_anchor"] for row in rows]),
        }

    conditional_advantages = [row["conditional_advantage"] for row in rows]
    anchor_advantages = [row["anchor_advantage"] for row in rows]
    anchor_unconditional_advantages = [row["anchor_vs_unconditional"] for row in rows]
    all_alpha_zero = all(row["alpha_zero_exact_all"] for row in rows)
    all_endpoints = all(row["all_alpha_one_endpoints"] and all(row[f"{condition}_endpoint_alpha_one"] for condition in CONDITIONS) for row in rows)
    all_finite = all(finite(row[f"{condition}_loss"]) for row in rows for condition in CONDITIONS)
    test_isolation = all(row["fresh_test_loaded_after_all_conditions"] for row in rows)
    conditional_wins = sum(value > 0 for value in conditional_advantages)
    anchor_wins = sum(value > 0 for value in anchor_advantages)
    anchor_unconditional_wins = sum(value > 0 for value in anchor_unconditional_advantages)
    conditional_summary = mean_sd(conditional_advantages)
    anchor_summary = mean_sd(anchor_advantages)
    anchor_unconditional_summary = mean_sd(anchor_unconditional_advantages)
    anchored_teacher_gap = summaries["conditional_downstream_anchor"]["teacher_gap"]["mean"]
    criteria = {
        "alpha_zero_exact_all": all_alpha_zero,
        "all_alpha_one_endpoints": all_endpoints,
        "all_fresh_test_losses_finite": all_finite,
        "fresh_test_loaded_after_all_conditions": test_isolation,
        "conditional_transport_beats_unconditional_in_at_least_4_of_5": conditional_wins >= 4,
        "conditional_transport_mean_advantage_at_least_0_05": conditional_summary["mean"] >= 0.05,
        "downstream_anchor_beats_conditional_in_at_least_4_of_5": anchor_wins >= 4,
        "downstream_anchor_mean_advantage_at_least_0_03": anchor_summary["mean"] >= 0.03,
        "downstream_anchor_within_0_15_of_teacher": anchored_teacher_gap <= 0.15,
    }
    criteria["third_layer_authorized"] = all(criteria.values())

    aggregate = {
        "experiment": "experiment_018_residual_interface_transport",
        "seeds": list(SEEDS),
        "rows": rows,
        "summaries": summaries,
        "paired_comparisons": {
            "conditional_minus_unconditional": {"wins": conditional_wins, **conditional_summary},
            "anchor_minus_conditional": {"wins": anchor_wins, **anchor_summary},
            "anchor_minus_unconditional": {"wins": anchor_unconditional_wins, **anchor_unconditional_summary},
        },
        "criteria": criteria,
    }
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    def f(value: float) -> str:
        return f"{value:.4f}"

    markdown = [
        "# Experiment 018 Aggregate Results",
        "",
        "**Scope:** Five paired seeds, three equal-parameter residual-interface transport conditions, and a fresh held-out WikiText-2 test slice (eligible test sequences 257–384). The final split was loaded only after all conditions completed their optimization in each seed.",
        "",
        "## Fresh-Test Loss by Seed",
        "",
        "| Seed | Teacher | Unconditional CE | Conditional CE | Conditional anchor | Conditional advantage (U − C) | Anchor advantage (C − A) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['seed']} | {f(row['teacher_loss'])} | {f(row['unconditional_ce_transport_loss'])} | {f(row['conditional_ce_transport_loss'])} | {f(row['conditional_downstream_anchor_loss'])} | {f(row['conditional_advantage'])} | {f(row['anchor_advantage'])} |"
        )
    markdown.extend([
        f"| **Mean ± sample SD** | **{f(summaries['teacher']['mean'])} ± {f(summaries['teacher']['sample_sd'])}** | **{f(summaries['unconditional_ce_transport']['fresh_test_loss']['mean'])} ± {f(summaries['unconditional_ce_transport']['fresh_test_loss']['sample_sd'])}** | **{f(summaries['conditional_ce_transport']['fresh_test_loss']['mean'])} ± {f(summaries['conditional_ce_transport']['fresh_test_loss']['sample_sd'])}** | **{f(summaries['conditional_downstream_anchor']['fresh_test_loss']['mean'])} ± {f(summaries['conditional_downstream_anchor']['fresh_test_loss']['sample_sd'])}** | **{f(conditional_summary['mean'])} ± {f(conditional_summary['sample_sd'])}** | **{f(anchor_summary['mean'])} ± {f(anchor_summary['sample_sd'])}** |",
        "",
        "A positive conditional advantage favors the conditional transport map. A positive anchor advantage favors the downstream-anchor condition.",
        "",
        "## Development Diagnostics after Layer 1",
        "",
        "| Condition | Development CE | Token-normalized source-logit KL | Correction RMS / Mamba RMS | Downstream-anchor diagnostic |",
        "|---|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        summary = summaries[condition]
        anchor = "—" if summary["development_l1_anchor"] is None else f"{f(summary['development_l1_anchor']['mean'])} ± {f(summary['development_l1_anchor']['sample_sd'])}"
        markdown.append(
            f"| {LABELS[condition].replace(chr(10), ' ')} | {f(summary['development_l1_ce']['mean'])} ± {f(summary['development_l1_ce']['sample_sd'])} | {f(summary['development_l1_token_kl']['mean'])} ± {f(summary['development_l1_token_kl']['sample_sd'])} | {f(summary['development_l1_correction_ratio']['mean'])} ± {f(summary['development_l1_correction_ratio']['sample_sd'])} | {anchor} |"
        )
    markdown.extend([
        "",
        "## Predeclared Criteria",
        "",
        "| Criterion | Result | Assessment |",
        "|---|---:|---|",
        f"| Exact alpha-zero source preservation | 5/5 seeds | {'**Pass**' if criteria['alpha_zero_exact_all'] else '**Fail**'} |",
        f"| Complete two-layer alpha-one endpoints | 5/5 seeds | {'**Pass**' if criteria['all_alpha_one_endpoints'] else '**Fail**'} |",
        f"| Fresh-test isolation after all conditions train | 5/5 seeds | {'**Pass**' if criteria['fresh_test_loaded_after_all_conditions'] else '**Fail**'} |",
        f"| Conditional CE wins versus unconditional CE | {conditional_wins}/5; mean advantage {f(conditional_summary['mean'])} | {'**Pass**' if criteria['conditional_transport_beats_unconditional_in_at_least_4_of_5'] and criteria['conditional_transport_mean_advantage_at_least_0_05'] else '**Fail**'} |",
        f"| Downstream anchor wins versus conditional CE | {anchor_wins}/5; mean advantage {f(anchor_summary['mean'])} | {'**Pass**' if criteria['downstream_anchor_beats_conditional_in_at_least_4_of_5'] and criteria['downstream_anchor_mean_advantage_at_least_0_03'] else '**Fail**'} |",
        f"| Anchor mean gap to frozen teacher <= 0.15 | {f(anchored_teacher_gap)} | {'**Pass**' if criteria['downstream_anchor_within_0_15_of_teacher'] else '**Fail**'} |",
        f"| Permission for third layer | — | {'**Authorized**' if criteria['third_layer_authorized'] else '**Denied**'} |",
        "",
        "## Fixed-Prompt Continuations",
        "",
    ])
    for row in rows:
        seed_item = raw[str(row["seed"])]
        markdown.append(f"### Seed {row['seed']}")
        markdown.append("")
        markdown.append("| Condition | Scientific-prompt continuation |")
        markdown.append("|---|---|")
        prompt = "The purpose of scientific research is to"
        for condition in CONDITIONS:
            continuation = seed_item[condition]["generation"][prompt].replace("|", "\\|")
            markdown.append(f"| {LABELS[condition].replace(chr(10), ' ')} | {continuation} |")
        markdown.append("")
    (OUT / "aggregate_results.md").write_text("\n".join(markdown) + "\n")

    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for row in rows:
        values = [row[f"{condition}_loss"] for condition in CONDITIONS]
        ax.plot(x, values, color="#94A3B8", linewidth=1.0, alpha=0.75, zorder=1)
        ax.scatter(x, values, color=[COLORS[c] for c in CONDITIONS], s=33, zorder=2)
    means = [summaries[c]["fresh_test_loss"]["mean"] for c in CONDITIONS]
    sds = [summaries[c]["fresh_test_loss"]["sample_sd"] for c in CONDITIONS]
    ax.errorbar(x, means, yerr=sds, color="#111827", linewidth=2.0, marker="D", markersize=6, capsize=4, label="mean ± sample SD", zorder=3)
    ax.axhline(summaries["teacher"]["mean"], color="#DC2626", linestyle="--", linewidth=1.5, label="frozen teacher")
    ax.set_xticks(x, [LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Fresh held-out next-token loss (lower is better)")
    ax.set_title("Experiment 018: matched transport endpoints")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    paired = [conditional_advantages, anchor_advantages]
    labels = ["Conditional −\nunconditional", "Anchor −\nconditional"]
    for index, values in enumerate(paired):
        jitter = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=38, color="#2563EB" if index == 0 else "#059669", zorder=2)
        ax.errorbar(index, statistics.mean(values), yerr=statistics.stdev(values), color="#111827", marker="D", markersize=6, capsize=4, zorder=3)
    ax.axhline(0.0, color="#374151", linewidth=1.1)
    ax.axhline(0.05, color="#2563EB", linestyle="--", linewidth=1.0, alpha=0.8, label="conditional criterion: +0.05")
    ax.axhline(0.03, color="#059669", linestyle=":", linewidth=1.2, alpha=0.9, label="anchor criterion: +0.03")
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Paired fresh-test advantage in loss (positive favors named condition)")
    ax.set_title("Predeclared causal comparisons")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fresh_test_transport_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
