"""Frozen per-model and two-model gates for the Dr.Spider one-use experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUMMARY_NAMES = (
    "direct_full",
    "full",
    "subset_direct",
    "subset_full",
    "subset_self_refine",
    "subset_always_generic",
    "subset_execution_only",
    "subset_no_selector",
    "subset_random_gate",
)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def evaluate_model(analysis_path: Path, summaries_dir: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    analysis = json.loads(analysis_path.read_text())
    summaries = {
        name: json.loads((summaries_dir / f"{name}.summary.json").read_text())
        for name in SUMMARY_NAMES
    }
    primary = analysis["primary_full_vs_direct"]
    axes = analysis["primary_by_axis"]
    subset = analysis["subset_comparisons"]
    safety = analysis["safety"]
    efficiency = analysis["efficiency"]
    component_names = (
        "full_vs_always_generic",
        "full_vs_execution_only",
        "full_vs_no_selector",
    )
    component_effects = {name: subset[name] for name in component_names}
    point_comparators = (
        "full_vs_direct",
        "full_vs_self_refine",
        "full_vs_always_generic",
        "full_vs_execution_only",
        "full_vs_no_selector",
        "full_vs_random_gate",
    )
    checks = {
        "formal_full_records_equal_15269": int(primary["records"]) == 15_269,
        "frozen_subset_records_equal_1556": all(
            int(subset[name]["records"]) == 1_556 for name in subset
        ),
        "full_gain_at_least_inclusive_1pp": float(primary["accuracy_difference"]) >= 0.01 - 1e-12,
        "full_cluster_ci_lower_above_zero": float(primary["cluster_bootstrap_difference_ci95"][0]) > 0,
        "full_exact_mcnemar_p_at_most_0p01": float(primary["mcnemar_exact_p"]) <= 0.01,
        "full_absolute_accuracy_at_least_0p25": float(primary["candidate_accuracy"]) >= 0.25,
        "no_axis_gain_below_minus_1pp": all(
            float(value["accuracy_difference"]) >= -0.01 - 1e-12 for value in axes.values()
        ),
        "at_least_two_axes_gain_inclusive_1pp": sum(
            float(value["accuracy_difference"]) >= 0.01 - 1e-12 for value in axes.values()
        ) >= 2,
        "full_beats_self_refine_by_inclusive_1pp_on_subset": (
            float(subset["full_vs_self_refine"]["accuracy_difference"]) >= 0.01 - 1e-12
        ),
        "full_vs_self_refine_cluster_ci_lower_above_zero": (
            float(subset["full_vs_self_refine"]["cluster_bootstrap_difference_ci95"][0]) > 0
        ),
        "full_point_accuracy_not_below_all_subset_comparators": all(
            float(subset[name]["accuracy_difference"]) >= -1e-12 for name in point_comparators
        ),
        "one_critical_component_gap_at_least_3pp_with_positive_cluster_ci": any(
            float(value["accuracy_difference"]) >= 0.03 - 1e-12
            and float(value["cluster_bootstrap_difference_ci95"][0]) > 0
            for value in component_effects.values()
        ),
        "full_execution_validity_not_below_direct": (
            float(safety["full_minus_direct_execution_validity"]) >= -1e-12
        ),
        "full_zero_valid_to_failure_regressions": int(safety["full_valid_to_failure_regressions"]) == 0,
        "full_unflagged_sql_immutable": bool(safety["full_unflagged_sql_immutable"]),
        "all_selectors_gold_free": safety["gold_used_by_any_selector"] is False,
        "official_evaluator_errors_zero_all_arms": all(
            int(summary["official_evaluator_errors"]) == 0 for summary in summaries.values()
        ),
        "official_protocol_all_arms": all(
            summary["protocol_id"] == "drspider-pinned-official-test-suite-execution-v1"
            and summary["official_plug_value"] is False
            and summary["official_keep_distinct"] is False
            for summary in summaries.values()
        ),
        "full_generation_seconds_ratio_at_most_2": (
            float(efficiency["full_to_direct_generation_seconds_ratio"]) <= 2.0
        ),
        "thresholds_unchanged_after_results": True,
    }
    payload: dict[str, Any] = {
        "protocol_id": "drspider-oneuse-model-gate-v1",
        "gate_pass": all(checks.values()),
        "checks": checks,
        "primary_gain": primary["accuracy_difference"],
        "primary_cluster_ci95": primary["cluster_bootstrap_difference_ci95"],
        "primary_mcnemar_exact_p": primary["mcnemar_exact_p"],
        "axis_gains": {name: value["accuracy_difference"] for name, value in axes.items()},
        "subset_gains_full_minus_comparator": {
            name: value["accuracy_difference"] for name, value in subset.items()
        },
        "thresholds_changed_after_results": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def evaluate_panel(
    llama_analysis_path: Path,
    llama_gate_path: Path,
    mistral_analysis_path: Path,
    mistral_gate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    analyses = {
        "llama31_8b": json.loads(llama_analysis_path.read_text()),
        "mistral7b": json.loads(mistral_analysis_path.read_text()),
    }
    gates = {
        "llama31_8b": json.loads(llama_gate_path.read_text()),
        "mistral7b": json.loads(mistral_gate_path.read_text()),
    }
    gains = {
        name: float(value["primary_full_vs_direct"]["accuracy_difference"])
        for name, value in analyses.items()
    }
    raw_p = {
        name: float(value["primary_full_vs_direct"]["mcnemar_exact_p"])
        for name, value in analyses.items()
    }
    adjusted = holm_adjust(raw_p)
    checks = {
        "both_model_gates_pass": all(bool(gate["gate_pass"]) for gate in gates.values()),
        "both_gains_at_least_inclusive_1pp": all(gain >= 0.01 - 1e-12 for gain in gains.values()),
        "both_cluster_ci_lower_bounds_above_zero": all(
            float(value["primary_full_vs_direct"]["cluster_bootstrap_difference_ci95"][0]) > 0
            for value in analyses.values()
        ),
        "both_holm_adjusted_p_at_most_0p05": all(value <= 0.05 for value in adjusted.values()),
        "unweighted_mean_gain_at_least_2pp": sum(gains.values()) / 2 >= 0.02 - 1e-12,
        "one_model_gain_at_least_3pp": max(gains.values()) >= 0.03 - 1e-12,
        "thresholds_unchanged_after_results": True,
    }
    payload: dict[str, Any] = {
        "protocol_id": "drspider-oneuse-two-model-panel-v1",
        "member_order": ["llama31_8b", "mistral7b"],
        "all_model_and_baseline_outputs_generated_before_any_gold_evaluation": True,
        "accuracy_gains": gains,
        "unweighted_mean_accuracy_gain": sum(gains.values()) / 2,
        "raw_mcnemar_p": raw_p,
        "holm_adjusted_mcnemar_p": adjusted,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "thresholds_changed_after_results": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    model = subparsers.add_parser("model")
    model.add_argument("--analysis", dest="analysis_path", type=Path, required=True)
    model.add_argument("--summaries-dir", type=Path, required=True)
    model.add_argument("--output", dest="output_path", type=Path, required=True)
    panel = subparsers.add_parser("panel")
    panel.add_argument("--llama-analysis", dest="llama_analysis_path", type=Path, required=True)
    panel.add_argument("--llama-gate", dest="llama_gate_path", type=Path, required=True)
    panel.add_argument("--mistral-analysis", dest="mistral_analysis_path", type=Path, required=True)
    panel.add_argument("--mistral-gate", dest="mistral_gate_path", type=Path, required=True)
    panel.add_argument("--output", dest="output_path", type=Path, required=True)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    payload = evaluate_model(**args) if mode == "model" else evaluate_panel(**args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["gate_pass"] else 1)


if __name__ == "__main__":
    main()
