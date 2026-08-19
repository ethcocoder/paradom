"""Aggregate Experiment 013's independent-seed, untouched-test replication results."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("research/experiment_013_replication")
OUTPUT = Path("research/experiment_013_analysis")
REQUIRED_SEEDS = (20260821, 20260822, 20260823)
SCIENTIFIC_PROMPT = "The purpose of scientific research is to"
EXPLORER_PROMPT = "Once upon a time, a young explorer discovered"


def mean_std(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("At least two independent runs are required for a sample standard deviation")
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def finite(value: float) -> bool:
    return math.isfinite(value)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def read_seed(seed: int) -> dict:
    path = ROOT / f"seed_{seed}" / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    generations: list[dict] = []

    for seed in REQUIRED_SEEDS:
        result = read_seed(seed)
        teacher = result["teacher"]["loss"]
        direct = result["direct"]
        random = result["random"]
        direct_loss = direct["final_test"]["loss"]
        random_loss = random["final_test"]["loss"]
        direct_l1 = direct["stages_development"]["layer_1"]["local_after"]["normalized_mse"]
        random_l1 = random["stages_development"]["layer_1"]["local_after"]["normalized_mse"]
        alpha_zero = direct["alpha_zero_test"]["loss"]
        rows.append(
            {
                "seed": seed,
                "teacher_test_loss": teacher,
                "direct_test_loss": direct_loss,
                "random_test_loss": random_loss,
                "direct_teacher_gap": direct_loss - teacher,
                "random_teacher_gap": random_loss - teacher,
                "direct_minus_random": direct_loss - random_loss,
                "direct_layer1_nmse_dev": direct_l1,
                "random_layer1_nmse_dev": random_l1,
                "alpha_zero_loss": alpha_zero,
                "alpha_zero_abs_error": abs(alpha_zero - teacher),
                "alpha_zero_preserves_teacher": result["interpretation"]["alpha_zero_preserves_teacher_on_test"],
                "all_final_losses_finite": finite(direct_loss) and finite(random_loss),
                "random_below_3_5": random_loss < 3.5,
            }
        )
        generations.append(
            {
                "seed": seed,
                "direct_science": direct["generation"][SCIENTIFIC_PROMPT],
                "direct_explorer": direct["generation"][EXPLORER_PROMPT],
                "random_science": random["generation"][SCIENTIFIC_PROMPT],
                "random_explorer": random["generation"][EXPLORER_PROMPT],
            }
        )

    teacher_values = [r["teacher_test_loss"] for r in rows]
    direct_values = [r["direct_test_loss"] for r in rows]
    random_values = [r["random_test_loss"] for r in rows]
    paired_values = [r["direct_minus_random"] for r in rows]
    direct_gap_values = [r["direct_teacher_gap"] for r in rows]
    random_gap_values = [r["random_teacher_gap"] for r in rows]
    direct_l1_values = [r["direct_layer1_nmse_dev"] for r in rows]
    random_l1_values = [r["random_layer1_nmse_dev"] for r in rows]

    aggregate = {
        "experiment": "experiment_013_independent_seed_replication",
        "n_independent_seeds": len(rows),
        "seeds": list(REQUIRED_SEEDS),
        "split_policy": {
            "calibration": "WikiText-2 train (256 sequences)",
            "development": "WikiText-2 validation (16 sequences; diagnostics only)",
            "final_test": "WikiText-2 test (16 sequences; no fitting or intermediate selection)",
        },
        "metrics": {
            "teacher_test_loss": mean_std(teacher_values),
            "direct_test_loss": mean_std(direct_values),
            "random_test_loss": mean_std(random_values),
            "direct_teacher_gap": mean_std(direct_gap_values),
            "random_teacher_gap": mean_std(random_gap_values),
            "paired_direct_minus_random_loss": mean_std(paired_values),
            "direct_layer1_nmse_development": mean_std(direct_l1_values),
            "random_layer1_nmse_development": mean_std(random_l1_values),
        },
        "acceptance": {
            "all_alpha_zero_exact": all(r["alpha_zero_preserves_teacher"] for r in rows),
            "all_final_losses_finite": all(r["all_final_losses_finite"] for r in rows),
            "all_random_test_losses_below_3_5": all(r["random_below_3_5"] for r in rows),
            "random_beats_direct_in_every_seed": all(r["direct_minus_random"] > 0 for r in rows),
            "generation_requires_human_language_review": True,
        },
        "per_seed": rows,
        "generations": generations,
        "statistical_note": "The reported standard deviations are sample standard deviations across three independent seeds. The paired difference is descriptive; n=3 is not sufficient for a robust inferential claim.",
    }
    (OUTPUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    markdown = [
        "# Experiment 013: Aggregated Replication Results",
        "",
        "## Held-Out Test Loss by Seed",
        "",
        "| Seed | Teacher test loss | Direct Mamba test loss | Random Mamba test loss | Direct − random | Random − teacher | α=0 absolute error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        markdown.append(
            f"| {r['seed']} | {fmt(r['teacher_test_loss'])} | {fmt(r['direct_test_loss'])} | {fmt(r['random_test_loss'])} | {fmt(r['direct_minus_random'])} | {fmt(r['random_teacher_gap'])} | {r['alpha_zero_abs_error']:.2e} |"
        )

    d = aggregate["metrics"]
    markdown += [
        "",
        "## Mean ± Sample Standard Deviation",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Teacher final test loss | {fmt(d['teacher_test_loss']['mean'])} ± {fmt(d['teacher_test_loss']['sample_std'])} |",
        f"| Direct initialization final test loss | {fmt(d['direct_test_loss']['mean'])} ± {fmt(d['direct_test_loss']['sample_std'])} |",
        f"| Random initialization final test loss | {fmt(d['random_test_loss']['mean'])} ± {fmt(d['random_test_loss']['sample_std'])} |",
        f"| Direct − random loss (paired) | {fmt(d['paired_direct_minus_random_loss']['mean'])} ± {fmt(d['paired_direct_minus_random_loss']['sample_std'])} |",
        f"| Random − teacher loss | {fmt(d['random_teacher_gap']['mean'])} ± {fmt(d['random_teacher_gap']['sample_std'])} |",
        f"| Direct layer-1 NMSE (development) | {fmt(d['direct_layer1_nmse_development']['mean'])} ± {fmt(d['direct_layer1_nmse_development']['sample_std'])} |",
        f"| Random layer-1 NMSE (development) | {fmt(d['random_layer1_nmse_development']['mean'])} ± {fmt(d['random_layer1_nmse_development']['sample_std'])} |",
        "",
        "## Mechanical Acceptance Checks",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| α=0 preserves teacher exactly on final test in every seed | {aggregate['acceptance']['all_alpha_zero_exact']} |",
        f"| Both final hybrid losses are finite in every seed | {aggregate['acceptance']['all_final_losses_finite']} |",
        f"| Random hybrid final test loss < 3.5 in every seed | {aggregate['acceptance']['all_random_test_losses_below_3_5']} |",
        f"| Random initialization beats direct mapping in every seed | {aggregate['acceptance']['random_beats_direct_in_every_seed']} |",
        "",
        "## Fixed-Prompt Continuations for Human Review",
        "",
        "| Seed | Condition | Scientific research prompt | Explorer prompt |",
        "|---:|---|---|---|",
    ]
    for g in generations:
        for condition, science_key, explorer_key in (("Direct", "direct_science", "direct_explorer"), ("Random", "random_science", "random_explorer")):
            science = g[science_key].replace("|", "\\|")
            explorer = g[explorer_key].replace("|", "\\|")
            markdown.append(f"| {g['seed']} | {condition} | {science} | {explorer} |")
    markdown += [
        "",
        "The loss comparison is a three-seed descriptive replication rather than a powered significance test. English quality must be judged from the displayed continuations rather than inferred from loss alone.",
        "",
    ]
    (OUTPUT / "aggregate_results.md").write_text("\n".join(markdown))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    x = list(range(len(rows)))
    ax.plot(x, teacher_values, color="#4C566A", marker="o", linewidth=2.4, label="Teacher (unchanged)")
    ax.plot(x, direct_values, color="#BF616A", marker="o", linewidth=2.4, label="Direct static map + distillation")
    ax.plot(x, random_values, color="#2E8B57", marker="o", linewidth=2.4, label="Random Mamba + distillation")
    ax.set_xticks(x, [str(r["seed"]) for r in rows])
    ax.set_ylabel("Final WikiText-2 test loss (lower is better)")
    ax.set_xlabel("Independent random seed")
    ax.set_title("Experiment 013: Two-Layer Attention-to-Mamba Functional Distillation")
    ax.legend(frameon=True, loc="upper right")
    ax.set_ylim(bottom=min(teacher_values + direct_values + random_values) - 0.1)
    fig.savefig(OUTPUT / "held_out_test_loss_by_seed.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"output": str(OUTPUT), "acceptance": aggregate["acceptance"], "metrics": aggregate["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
