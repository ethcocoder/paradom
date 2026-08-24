from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("research/experiment_020a_relational_geometry_pilot")
OUT = Path("research/experiment_020a_relational_geometry_analysis")
SEEDS = (20260881, 20260882, 20260883, 20260884, 20260885)
CONTROL = "conditional_ce_transport"
CANDIDATES = ("relational_geometry_lambda_005", "relational_geometry_lambda_010", "relational_geometry_lambda_020")
COEFFICIENTS = {"relational_geometry_lambda_005": 0.05, "relational_geometry_lambda_010": 0.10, "relational_geometry_lambda_020": 0.20}
COLORS = {"relational_geometry_lambda_005": "#2563EB", "relational_geometry_lambda_010": "#7C3AED", "relational_geometry_lambda_020": "#0F766E"}


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": float(statistics.mean(values)), "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0}


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int) or isinstance(value, bool) or value is None or isinstance(value, str):
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
            raise FileNotFoundError(f"Missing pilot output: {path}")
        item = json.loads(path.read_text())
        if item["data"]["test_split_requested"] is not False:
            raise RuntimeError(f"Seed {seed} violated the test-split prohibition")
        raw[str(seed)] = item
        control_endpoint = item[CONTROL]["development_endpoint"]
        row = {"seed": seed, "test_split_requested": item["data"]["test_split_requested"]}
        for condition in (CONTROL,) + CANDIDATES:
            endpoint = item[condition]["development_endpoint"]
            integrity = item[condition]["alpha_zero_integrity"]
            row[f"{condition}_ce"] = endpoint["development_ce"]
            row[f"{condition}_relation"] = endpoint["post_block_relation_kl"]
            row[f"{condition}_drift_l1"] = endpoint["post_layer_1_relative_drift"]
            row[f"{condition}_drift_l2"] = endpoint["post_layer_2_relative_drift"]
            row[f"{condition}_alpha_zero_deviation"] = integrity["absolute_loss_deviation"]
            row[f"{condition}_endpoint"] = item[condition]["endpoint_alpha_one"]
            row[f"{condition}_finite"] = finite_tree(item[condition]["development_endpoint"])
            if condition != CONTROL:
                row[f"{condition}_ce_delta"] = endpoint["development_ce"] - control_endpoint["development_ce"]
                row[f"{condition}_relation_delta"] = endpoint["post_block_relation_kl"] - control_endpoint["post_block_relation_kl"]
        rows.append(row)

    candidates: dict[str, Any] = {}
    for condition in CANDIDATES:
        ce_delta = [row[f"{condition}_ce_delta"] for row in rows]
        relation_delta = [row[f"{condition}_relation_delta"] for row in rows]
        alpha_zero_max = max(row[f"{condition}_alpha_zero_deviation"] for row in rows)
        finite = all(row[f"{condition}_finite"] for row in rows)
        endpoints = all(row[f"{condition}_endpoint"] for row in rows)
        mean_ce = mean_sd(ce_delta)
        mean_relation = mean_sd(relation_delta)
        eligible = finite and endpoints and alpha_zero_max <= 1e-5 and mean_ce["mean"] <= 0.02 and mean_relation["mean"] < 0.0
        candidates[condition] = {
            "coefficient": COEFFICIENTS[condition],
            "development_ce_delta_vs_control": mean_ce,
            "relation_kl_delta_vs_control": mean_relation,
            "maximum_alpha_zero_loss_deviation": alpha_zero_max,
            "all_endpoint_metrics_finite": finite,
            "all_endpoints_alpha_one": endpoints,
            "eligible_under_locked_rule": eligible,
            "development_ce_by_condition": mean_sd([row[f"{condition}_ce"] for row in rows]),
            "relation_kl_by_condition": mean_sd([row[f"{condition}_relation"] for row in rows]),
            "post_layer_2_drift_by_condition": mean_sd([row[f"{condition}_drift_l2"] for row in rows]),
        }

    eligible = [item for item in candidates.values() if item["eligible_under_locked_rule"]]
    selected = max(eligible, key=lambda item: item["coefficient"]) if eligible else None
    aggregate = {
        "experiment": "experiment_020a_relational_geometry_pilot",
        "scope": "development_only_no_test_access",
        "seeds": list(SEEDS),
        "test_split_requested_by_all_seeds": all(row["test_split_requested"] is False for row in rows),
        "control": {
            "development_ce": mean_sd([row[f"{CONTROL}_ce"] for row in rows]),
            "post_block_relation_kl": mean_sd([row[f"{CONTROL}_relation"] for row in rows]),
            "post_layer_1_relative_drift": mean_sd([row[f"{CONTROL}_drift_l1"] for row in rows]),
            "post_layer_2_relative_drift": mean_sd([row[f"{CONTROL}_drift_l2"] for row in rows]),
            "maximum_alpha_zero_loss_deviation": max(row[f"{CONTROL}_alpha_zero_deviation"] for row in rows),
        },
        "candidates": candidates,
        "locked_selection": {
            "selected_coefficient": None if selected is None else selected["coefficient"],
            "selected_condition": None if selected is None else next(key for key, value in candidates.items() if value is selected),
            "endpoint_experiment_authorized": selected is not None,
            "selection_rule": "largest coefficient satisfying all finite endpoints, all alpha-one endpoints, max alpha-zero loss deviation <=1e-5, mean development CE increase <=0.020, and mean relation KL lower than CE control",
        },
        "rows": rows,
    }
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = [
        "# Experiment 020A Development-Only Aggregate",
        "",
        "**Scope:** Five-seed coefficient-selection pilot. The script loaded only WikiText-2 train and validation data; every result explicitly records `test_split_requested: false`. These are development diagnostics, not held-out transfer results.",
        "",
        "## Per-Seed Development Differences",
        "",
        "Positive CE delta is worse than the conditional-CE transport control. Negative relation delta indicates lower post-block relation KL than that control.",
        "",
        "| Seed | λ=0.05 CE Δ | λ=0.10 CE Δ | λ=0.20 CE Δ | λ=0.05 relation Δ | λ=0.10 relation Δ | λ=0.20 relation Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['seed']} | {fmt(row['relational_geometry_lambda_005_ce_delta'])} | {fmt(row['relational_geometry_lambda_010_ce_delta'])} | {fmt(row['relational_geometry_lambda_020_ce_delta'])} | {fmt(row['relational_geometry_lambda_005_relation_delta'])} | {fmt(row['relational_geometry_lambda_010_relation_delta'])} | {fmt(row['relational_geometry_lambda_020_relation_delta'])} |")
    lines.extend(["", "## Locked Coefficient Selection", "", "| Coefficient | Mean development CE Δ ± sample SD | Mean relation-KL Δ ± sample SD | Maximum alpha-zero loss deviation | All finite / alpha-one | Eligible |", "|---:|---:|---:|---:|---|---|"])
    for condition in CANDIDATES:
        item = candidates[condition]
        ce = item["development_ce_delta_vs_control"]
        relation = item["relation_kl_delta_vs_control"]
        integrity = "yes" if item["all_endpoint_metrics_finite"] and item["all_endpoints_alpha_one"] else "no"
        eligibility = "**yes**" if item["eligible_under_locked_rule"] else "no"
        lines.append(f"| {item['coefficient']:.2f} | {fmt(ce['mean'])} ± {fmt(ce['sample_sd'])} | {fmt(relation['mean'])} ± {fmt(relation['sample_sd'])} | {item['maximum_alpha_zero_loss_deviation']:.2e} | {integrity} | {eligibility} |")
    selection_text = "No coefficient was eligible; endpoint work is cancelled." if selected is None else f"**Selected coefficient: λ={selected['coefficient']:.2f}.** It is the largest eligible coefficient under the locked development-only rule. This selection does not establish an endpoint benefit and does not access the test partition."
    lines.extend(["", selection_text, ""])
    (OUT / "aggregate_results.md").write_text("\n".join(lines))

    x = np.arange(len(CANDIDATES))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for axis, metric, ylabel, title in [
        (axes[0], "ce_delta", "Development CE difference versus CE control\n(positive is worse)", "Language objective safety diagnostic"),
        (axes[1], "relation_delta", "Post-block relation-KL difference versus CE control\n(negative is lower)", "Relational geometry mechanism diagnostic"),
    ]:
        for row in rows:
            values = [row[f"{condition}_{metric}"] for condition in CANDIDATES]
            axis.plot(x, values, color="#94A3B8", linewidth=1.0, alpha=0.8, zorder=1)
            axis.scatter(x, values, color=[COLORS[condition] for condition in CANDIDATES], s=38, zorder=2)
        means = [candidates[condition]["development_ce_delta_vs_control" if metric == "ce_delta" else "relation_kl_delta_vs_control"]["mean"] for condition in CANDIDATES]
        sds = [candidates[condition]["development_ce_delta_vs_control" if metric == "ce_delta" else "relation_kl_delta_vs_control"]["sample_sd"] for condition in CANDIDATES]
        axis.errorbar(x, means, yerr=sds, color="#111827", marker="D", linewidth=2, capsize=4, label="mean ± sample SD", zorder=3)
        axis.axhline(0, color="#374151", linestyle="--", linewidth=1)
        if metric == "ce_delta":
            axis.axhline(0.02, color="#DC2626", linestyle=":", linewidth=1.5, label="locked CE ceiling")
        axis.set_xticks(x, ["λ=0.05", "λ=0.10", "λ=0.20"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 020A: development-only relational-geometry coefficient pilot", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "coefficient_selection_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
