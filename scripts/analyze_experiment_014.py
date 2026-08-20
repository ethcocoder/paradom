"""Aggregate the locked five-seed causal-control results for Experiment 014."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("research/experiment_014_causal_control")
OUTPUT = Path("research/experiment_014_analysis")
SEEDS = (20260831, 20260832, 20260833, 20260834, 20260835)
SCIENCE_PROMPT = "The purpose of scientific research is to"
EXPLORER_PROMPT = "Once upon a time, a young explorer discovered"


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def bootstrap_mean_ci(values: list[float]) -> dict[str, float]:
    """Exact equal-probability bootstrap over all n^n resamples (n=5)."""
    n = len(values)
    means = sorted(sum(values[index] for index in sample) / n for sample in itertools.product(range(n), repeat=n))
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return {"percentile_95_lower": lower, "percentile_95_upper": upper, "resamples": len(means)}


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    generations: list[dict] = []

    for seed in SEEDS:
        path = ROOT / f"seed_{seed}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing locked result {path}")
        result = json.loads(path.read_text())
        functional = result["functional"]
        ce_only = result["ce_only"]
        teacher_loss = result["teacher"]["loss"]
        functional_loss = functional["final_test"]["loss"]
        ce_loss = ce_only["final_test"]["loss"]
        functional_l1_nmse = functional["stages_development"]["layer_1"]["local_diagnostic_post"]["normalized_mse"]
        ce_l1_nmse = ce_only["stages_development"]["layer_1"]["local_diagnostic_post"]["normalized_mse"]
        rows.append(
            {
                "seed": seed,
                "teacher_test_loss": teacher_loss,
                "functional_test_loss": functional_loss,
                "ce_only_test_loss": ce_loss,
                "ce_only_minus_functional": ce_loss - functional_loss,
                "functional_minus_teacher": functional_loss - teacher_loss,
                "ce_only_minus_teacher": ce_loss - teacher_loss,
                "functional_l1_nmse_development": functional_l1_nmse,
                "ce_only_l1_nmse_development": ce_l1_nmse,
                "functional_alpha_zero_error": abs(functional["alpha_zero_test"]["loss"] - teacher_loss),
                "ce_only_alpha_zero_error": abs(ce_only["alpha_zero_test"]["loss"] - teacher_loss),
                "finite": math.isfinite(functional_loss) and math.isfinite(ce_loss),
            }
        )
        generations.append(
            {
                "seed": seed,
                "functional_science": functional["generation"][SCIENCE_PROMPT],
                "functional_explorer": functional["generation"][EXPLORER_PROMPT],
                "ce_only_science": ce_only["generation"][SCIENCE_PROMPT],
                "ce_only_explorer": ce_only["generation"][EXPLORER_PROMPT],
            }
        )

    functional_values = [row["functional_test_loss"] for row in rows]
    ce_values = [row["ce_only_test_loss"] for row in rows]
    paired_advantage = [row["ce_only_minus_functional"] for row in rows]
    teacher_values = [row["teacher_test_loss"] for row in rows]
    functional_gap = [row["functional_minus_teacher"] for row in rows]
    ce_gap = [row["ce_only_minus_teacher"] for row in rows]
    functional_nmse = [row["functional_l1_nmse_development"] for row in rows]
    ce_nmse = [row["ce_only_l1_nmse_development"] for row in rows]

    aggregate = {
        "experiment": "experiment_014_causal_control",
        "n_independent_paired_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "split_policy": {
            "calibration": "WikiText-2 raw train, 1,024 sequences",
            "development": "WikiText-2 raw validation, 64 sequences; diagnostics only",
            "final_test": "WikiText-2 raw test, 128 sequences / 5,894 scoring tokens; no fitting or model selection",
        },
        "metrics": {
            "teacher_test_loss": summary(teacher_values),
            "functional_test_loss": summary(functional_values),
            "ce_only_test_loss": summary(ce_values),
            "paired_ce_only_minus_functional_loss": summary(paired_advantage),
            "paired_ce_only_minus_functional_loss_bootstrap": bootstrap_mean_ci(paired_advantage),
            "functional_teacher_gap": summary(functional_gap),
            "ce_only_teacher_gap": summary(ce_gap),
            "functional_layer1_nmse_development": summary(functional_nmse),
            "ce_only_layer1_nmse_development": summary(ce_nmse),
        },
        "acceptance": {
            "alpha_zero_exact_all_conditions_all_seeds": all(row["functional_alpha_zero_error"] < 1e-5 and row["ce_only_alpha_zero_error"] < 1e-5 for row in rows),
            "finite_final_losses_all_conditions_all_seeds": all(row["finite"] for row in rows),
            "functional_beats_ce_only_in_at_least_4_of_5": sum(value > 0 for value in paired_advantage) >= 4,
            "functional_mean_paired_advantage_at_least_0_05": statistics.mean(paired_advantage) >= 0.05,
            "functional_teacher_gap_no_worse_than_0_15": statistics.mean(functional_gap) <= 0.15,
            "functional_beats_ce_only_count": sum(value > 0 for value in paired_advantage),
            "ce_only_beats_functional_count": sum(value < 0 for value in paired_advantage),
            "generation_requires_human_review": True,
        },
        "per_seed": rows,
        "generations": generations,
        "interpretation_guardrail": "Positive CE-only minus functional loss favors functional distillation. This five-seed result is a matched causal test of teacher-derived optimization signals, not an end-to-end architecture-conversion result.",
    }
    (OUTPUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    table = [
        "# Experiment 014: Five-Seed Causal-Control Aggregate",
        "",
        "## Frozen Held-Out Test Results",
        "",
        "| Seed | Teacher loss | Functional loss | CE-only loss | CE-only − functional | Functional − teacher |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['seed']} | {fmt(row['teacher_test_loss'])} | {fmt(row['functional_test_loss'])} | {fmt(row['ce_only_test_loss'])} | {fmt(row['ce_only_minus_functional'])} | {fmt(row['functional_minus_teacher'])} |"
        )
    metrics = aggregate["metrics"]
    table += [
        "",
        "## Mean ± Sample Standard Deviation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Teacher test loss | {fmt(metrics['teacher_test_loss']['mean'])} ± {fmt(metrics['teacher_test_loss']['sample_std'])} |",
        f"| Functional-distillation test loss | {fmt(metrics['functional_test_loss']['mean'])} ± {fmt(metrics['functional_test_loss']['sample_std'])} |",
        f"| CE-only test loss | {fmt(metrics['ce_only_test_loss']['mean'])} ± {fmt(metrics['ce_only_test_loss']['sample_std'])} |",
        f"| CE-only − functional loss | {fmt(metrics['paired_ce_only_minus_functional_loss']['mean'])} ± {fmt(metrics['paired_ce_only_minus_functional_loss']['sample_std'])} |",
        f"| Bootstrap 95% interval for paired difference | [{fmt(metrics['paired_ce_only_minus_functional_loss_bootstrap']['percentile_95_lower'])}, {fmt(metrics['paired_ce_only_minus_functional_loss_bootstrap']['percentile_95_upper'])}] |",
        f"| Functional layer-1 NMSE (development) | {fmt(metrics['functional_layer1_nmse_development']['mean'])} ± {fmt(metrics['functional_layer1_nmse_development']['sample_std'])} |",
        f"| CE-only layer-1 NMSE (development, post-hoc only) | {fmt(metrics['ce_only_layer1_nmse_development']['mean'])} ± {fmt(metrics['ce_only_layer1_nmse_development']['sample_std'])} |",
        "",
        "A positive paired value favors functional distillation. The bootstrap interval is descriptive because the experiment has only five seeds.",
        "",
        "## Fixed-Prompt Continuations for Human Review",
        "",
        "| Seed | Condition | Scientific-research prompt | Explorer prompt |",
        "|---:|---|---|---|",
    ]
    for generation in generations:
        for name, science_key, explorer_key in (("Functional", "functional_science", "functional_explorer"), ("CE-only", "ce_only_science", "ce_only_explorer")):
            science = generation[science_key].replace("|", "\\|").replace("\n", " ")
            explorer = generation[explorer_key].replace("|", "\\|").replace("\n", " ")
            table.append(f"| {generation['seed']} | {name} | {science} | {explorer} |")
    (OUTPUT / "aggregate_results.md").write_text("\n".join(table) + "\n")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    positions = range(len(SEEDS))
    for index, row in enumerate(rows):
        ax.plot([index - 0.13, index + 0.13], [row["functional_test_loss"], row["ce_only_test_loss"]], color="#A3BE8C", linewidth=1.5, zorder=1)
    ax.scatter([index - 0.13 for index in positions], functional_values, color="#5E81AC", s=65, zorder=2, label="Functional distillation")
    ax.scatter([index + 0.13 for index in positions], ce_values, color="#D08770", s=65, zorder=2, label="CE-only control")
    ax.axhline(statistics.mean(teacher_values), color="#4C566A", linewidth=2, linestyle="--", label="Frozen teacher")
    ax.set_xticks(list(positions), [str(seed) for seed in SEEDS])
    ax.set_xlabel("Paired independent seed")
    ax.set_ylabel("Final held-out WikiText-2 test loss (lower is better)")
    ax.set_title("Experiment 014: Teacher-Guided Functional Distillation vs CE-Only")
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(bottom=min(functional_values + ce_values + teacher_values) - 0.1)
    fig.savefig(OUTPUT / "paired_held_out_test_loss.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"acceptance": aggregate["acceptance"], "metrics": aggregate["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
