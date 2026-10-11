"""Analyze Dr.Spider OOD arms with original-question cluster uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .paired_statistics import exact_mcnemar_p
from .run_spider_feedback import load_jsonl


ARM_NAMES = (
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


def _keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    keyed = {str(row["case_id"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate case identifier in {path}")
    return keyed


def cluster_paired_statistics(
    direct_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    seed: int = 20260822,
    samples: int = 50_000,
) -> dict[str, Any]:
    direct = {str(row["case_id"]): row for row in direct_rows}
    candidate = {str(row["case_id"]): row for row in candidate_rows}
    if set(direct) != set(candidate):
        raise ValueError("paired Dr.Spider arms have different case identifiers")
    case_ids = sorted(direct)
    if not case_ids:
        raise ValueError("paired Dr.Spider comparison is empty")
    left = np.array([bool(direct[key]["correct"]) for key in case_ids], dtype=bool)
    right = np.array([bool(candidate[key]["correct"]) for key in case_ids], dtype=bool)
    clusters = np.array([int(direct[key]["q_id_spider_dev"]) for key in case_ids])
    if any(
        int(candidate[key]["q_id_spider_dev"]) != cluster
        for key, cluster in zip(case_ids, clusters, strict=True)
    ):
        raise ValueError("paired Dr.Spider arms disagree on original question cluster")
    unique = np.unique(clusters)
    cluster_n = np.array([np.sum(clusters == value) for value in unique], dtype=np.int64)
    case_delta = right.astype(np.int8) - left.astype(np.int8)
    cluster_delta = np.array(
        [np.sum(case_delta[clusters == value]) for value in unique], dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    chunk = 1_000
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        indices = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        numerator = cluster_delta[indices].sum(axis=1)
        denominator = cluster_n[indices].sum(axis=1)
        boot[start:stop] = numerator / denominator
    wins = int(np.sum(~left & right))
    losses = int(np.sum(left & ~right))
    return {
        "records": len(case_ids),
        "original_question_clusters": len(unique),
        "direct_accuracy": float(np.mean(left)),
        "candidate_accuracy": float(np.mean(right)),
        "accuracy_difference": float(np.mean(right) - np.mean(left)),
        "cluster_bootstrap_difference_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "candidate_only_correct": wins,
        "direct_only_correct": losses,
        "mcnemar_exact_p": exact_mcnemar_p(wins, losses),
        "bootstrap_unit": "q_id_spider_dev_original_question",
        "bootstrap_samples": samples,
        "seed": seed,
    }


def _filter(rows: dict[str, dict[str, Any]], field: str, value: str) -> list[dict[str, Any]]:
    return [row for row in rows.values() if str(row[field]) == value]


def _generation_totals(rows: list[dict[str, Any]], ids: set[str] | None = None) -> dict[str, float | int]:
    selected = rows if ids is None else [row for row in rows if str(row["case_id"]) in ids]
    return {
        "records": len(selected),
        "generation_seconds": sum(float(row["generation_latency_seconds"]) for row in selected),
        "input_tokens": sum(int(row["input_tokens"]) for row in selected),
        "output_tokens": sum(int(row["output_tokens"]) for row in selected),
    }


def analyze(
    cases_dir: Path,
    arms_dir: Path,
    initial_path: Path,
    generic_path: Path,
    self_refine_feedback_path: Path,
    self_refine_iterate_path: Path,
    arm_audit_path: Path,
    output_path: Path,
    seed: int = 20260822,
    bootstrap_samples: int = 50_000,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    cases = {
        name: _keyed(cases_dir / f"{name}.cases.jsonl") for name in ARM_NAMES
    }
    arms = {name: _keyed(arms_dir / f"{name}.jsonl") for name in ARM_NAMES}
    for name in ARM_NAMES:
        if set(cases[name]) != set(arms[name]):
            raise ValueError(f"case/prediction identifiers differ for {name}")

    primary = cluster_paired_statistics(
        list(cases["direct_full"].values()), list(cases["full"].values()), seed, bootstrap_samples
    )
    by_axis = {
        axis: cluster_paired_statistics(
            _filter(cases["direct_full"], "axis", axis),
            _filter(cases["full"], "axis", axis),
            seed + index + 1,
            bootstrap_samples,
        )
        for index, axis in enumerate(("DB", "NLQ", "SQL"))
    }
    subset_comparators = {
        "direct": "subset_direct",
        "self_refine": "subset_self_refine",
        "always_generic": "subset_always_generic",
        "execution_only": "subset_execution_only",
        "no_selector": "subset_no_selector",
        "random_gate": "subset_random_gate",
    }
    subset = {
        f"full_vs_{label}": cluster_paired_statistics(
            list(cases[arm].values()),
            list(cases["subset_full"].values()),
            seed + 10 + index,
            bootstrap_samples,
        )
        for index, (label, arm) in enumerate(subset_comparators.items())
    }

    direct_predictions = arms["direct_full"]
    full_predictions = arms["full"]
    ids = set(direct_predictions)
    unflagged = {case_id for case_id in ids if not bool(direct_predictions[case_id]["needs_repair"])}
    direct_valid = np.array([bool(direct_predictions[key]["execution_valid"]) for key in sorted(ids)])
    full_valid = np.array([bool(full_predictions[key]["execution_valid"]) for key in sorted(ids)])

    raw_initial = load_jsonl(initial_path)
    raw_generic = load_jsonl(generic_path)
    raw_feedback = load_jsonl(self_refine_feedback_path)
    raw_iterate = load_jsonl(self_refine_iterate_path)
    flagged_ids = {
        str(row["case_id"]) for row in raw_initial if bool(row["needs_repair"])
    }
    subset_ids = set(cases["subset_direct"])
    direct_cost = _generation_totals(raw_initial)
    full_repair_cost = _generation_totals(raw_generic, flagged_ids)
    full_cost = {
        key: direct_cost[key] + full_repair_cost[key]
        for key in ("records", "generation_seconds", "input_tokens", "output_tokens")
    }
    subset_initial_cost = _generation_totals(raw_initial, subset_ids)
    feedback_cost = _generation_totals(raw_feedback)
    iterate_cost = _generation_totals(raw_iterate)
    self_refine_cost = {
        key: subset_initial_cost[key] + feedback_cost[key] + iterate_cost[key]
        for key in ("records", "generation_seconds", "input_tokens", "output_tokens")
    }
    arm_audit = json.loads(arm_audit_path.read_text())
    payload: dict[str, Any] = {
        "protocol_id": "drspider-oneuse-cluster-paired-analysis-v1",
        "primary_full_vs_direct": primary,
        "primary_by_axis": by_axis,
        "subset_comparisons": subset,
        "safety": {
            "direct_execution_validity": float(np.mean(direct_valid)),
            "full_execution_validity": float(np.mean(full_valid)),
            "full_minus_direct_execution_validity": float(np.mean(full_valid) - np.mean(direct_valid)),
            "full_valid_to_failure_regressions": int(np.sum(direct_valid & ~full_valid)),
            "full_unflagged_sql_immutable": all(
                full_predictions[key]["sql"] == direct_predictions[key]["sql"] for key in unflagged
            ),
            "gold_used_by_any_selector": bool(arm_audit["gold_used_by_any_selector"]),
            "critic_flagged_full": len(flagged_ids),
        },
        "efficiency": {
            "direct_full": direct_cost,
            "full_flagged_repairs": full_repair_cost,
            "full_end_to_end": full_cost,
            "full_to_direct_generation_seconds_ratio": (
                float(full_cost["generation_seconds"]) / float(direct_cost["generation_seconds"])
            ),
            "full_to_direct_output_token_ratio": (
                float(full_cost["output_tokens"]) / float(direct_cost["output_tokens"])
            ),
            "subset_initial": subset_initial_cost,
            "subset_self_refine_feedback": feedback_cost,
            "subset_self_refine_iterate": iterate_cost,
            "subset_self_refine_end_to_end": self_refine_cost,
            "self_refine_to_subset_direct_generation_seconds_ratio": (
                float(self_refine_cost["generation_seconds"])
                / float(subset_initial_cost["generation_seconds"])
            ),
            "measurement_scope": "model_generate_wall_time_allocated_equally_within_each_batch; excludes model load and official evaluation",
        },
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--initial", dest="initial_path", type=Path, required=True)
    parser.add_argument("--generic", dest="generic_path", type=Path, required=True)
    parser.add_argument("--self-refine-feedback", dest="self_refine_feedback_path", type=Path, required=True)
    parser.add_argument("--self-refine-iterate", dest="self_refine_iterate_path", type=Path, required=True)
    parser.add_argument("--arm-audit", dest="arm_audit_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    args = parser.parse_args()
    print(json.dumps(analyze(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
