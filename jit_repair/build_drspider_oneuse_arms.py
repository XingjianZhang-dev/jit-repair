"""Build frozen gold-free Dr.Spider method, baseline, and ablation arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .drspider_oneuse import deterministic_rank, file_sha256
from .merge_spider_critic_retries import choose
from .run_spider_feedback import load_jsonl


ARM_FILES = {
    "direct_full": "direct_full.jsonl",
    "full": "full.jsonl",
    "subset_direct": "subset_direct.jsonl",
    "subset_full": "subset_full.jsonl",
    "subset_self_refine": "subset_self_refine.jsonl",
    "subset_always_generic": "subset_always_generic.jsonl",
    "subset_execution_only": "subset_execution_only.jsonl",
    "subset_no_selector": "subset_no_selector.jsonl",
    "subset_random_gate": "subset_random_gate.jsonl",
}


def _keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    keyed = {str(row["case_id"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate case identifier in {path}")
    return keyed


def _selected(
    row: dict[str, Any], arm: str, reason: str
) -> dict[str, Any]:
    value = dict(row)
    value["derived_arm"] = arm
    value["selection_reason"] = reason
    value["gold_used_by_selector"] = False
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(
    safe_path: Path,
    initial_path: Path,
    generic_path: Path,
    self_refine_path: Path,
    output_dir: Path,
    audit_output: Path,
    seed: int = 20260822,
) -> dict[str, Any]:
    planned = [output_dir / name for name in ARM_FILES.values()] + [audit_output]
    if any(path.exists() for path in planned):
        raise FileExistsError("refusing to overwrite Dr.Spider derived arm output")
    safe_rows = load_jsonl(safe_path)
    if len(safe_rows) != 15_269:
        raise ValueError("formal Dr.Spider arm build requires 15,269 safe rows")
    safe_ids = [str(row["case_id"]) for row in safe_rows]
    if len(set(safe_ids)) != len(safe_ids):
        raise ValueError("duplicate safe benchmark case identifier")
    subset_ids = [
        str(row["case_id"]) for row in safe_rows if bool(row["in_ablation_subset"])
    ]
    if len(subset_ids) != 1_556:
        raise ValueError("formal Dr.Spider ablation subset must contain 1,556 rows")
    subset_id_set = set(subset_ids)

    initial = _keyed(initial_path)
    generic = _keyed(generic_path)
    self_refine = _keyed(self_refine_path)
    if set(initial) != set(safe_ids):
        raise ValueError("initial predictions do not exactly cover the formal benchmark")
    expected_generic = {
        case_id
        for case_id in safe_ids
        if case_id in subset_id_set or bool(initial[case_id]["needs_repair"])
    }
    if set(generic) != expected_generic:
        raise ValueError("generic repairs do not exactly cover subset union flagged cases")
    if set(self_refine) != set(subset_ids):
        raise ValueError("Self-Refine outputs do not exactly cover the frozen subset")
    for source in (initial, generic, self_refine):
        if any(bool(row.get("gold_visible_to_generation", True)) for row in source.values()):
            raise ValueError("model output does not attest gold-blind generation")

    full: dict[str, dict[str, Any]] = {}
    for case_id in safe_ids:
        previous = initial[case_id]
        if not bool(previous["needs_repair"]):
            full[case_id] = _selected(previous, "full", "unflagged_initial_immutable")
        else:
            selected, reason = choose(previous, generic[case_id])
            full[case_id] = _selected(selected, "full", reason)

    flagged_subset = [case_id for case_id in subset_ids if bool(initial[case_id]["needs_repair"])]
    random_budget = len(flagged_subset)
    random_ids = set(
        sorted(
            subset_ids,
            key=lambda case_id: deterministic_rank(seed, "random_gate", case_id),
        )[:random_budget]
    )

    arms: dict[str, list[dict[str, Any]]] = {
        "direct_full": [
            _selected(initial[case_id], "direct_full", "direct_initial")
            for case_id in safe_ids
        ],
        "full": [full[case_id] for case_id in safe_ids],
        "subset_direct": [
            _selected(initial[case_id], "subset_direct", "direct_initial")
            for case_id in subset_ids
        ],
        "subset_full": [
            _selected(full[case_id], "subset_full", str(full[case_id]["selection_reason"]))
            for case_id in subset_ids
        ],
        "subset_self_refine": [
            _selected(self_refine[case_id], "subset_self_refine", "official_pattern_iterate_output")
            for case_id in subset_ids
        ],
        "subset_always_generic": [
            _selected(generic[case_id], "subset_always_generic", "always_use_generic_repair")
            for case_id in subset_ids
        ],
        "subset_execution_only": [],
        "subset_no_selector": [],
        "subset_random_gate": [],
    }
    for case_id in subset_ids:
        previous, repair = initial[case_id], generic[case_id]
        if not bool(previous["execution_valid"]):
            selected, reason = choose(previous, repair)
            arms["subset_execution_only"].append(
                _selected(selected, "subset_execution_only", "execution_only_gate:" + reason)
            )
        else:
            arms["subset_execution_only"].append(
                _selected(previous, "subset_execution_only", "execution_valid_initial_immutable")
            )

        if bool(previous["needs_repair"]):
            arms["subset_no_selector"].append(
                _selected(repair, "subset_no_selector", "critic_gate_blindly_accepts_repair")
            )
        else:
            arms["subset_no_selector"].append(
                _selected(previous, "subset_no_selector", "unflagged_initial_immutable")
            )

        if case_id in random_ids:
            selected, reason = choose(previous, repair)
            arms["subset_random_gate"].append(
                _selected(selected, "subset_random_gate", "random_gate:" + reason)
            )
        else:
            arms["subset_random_gate"].append(
                _selected(previous, "subset_random_gate", "random_gate_not_selected")
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    for arm, rows in arms.items():
        _write_jsonl(output_dir / ARM_FILES[arm], rows)
    reason_counts: dict[str, dict[str, int]] = {}
    for arm, rows in arms.items():
        counts: dict[str, int] = {}
        for row in rows:
            reason = str(row["selection_reason"])
            counts[reason] = counts.get(reason, 0) + 1
        reason_counts[arm] = counts
    payload: dict[str, Any] = {
        "protocol_id": "drspider-gold-free-derived-arms-v1",
        "records_full": len(safe_ids),
        "records_subset": len(subset_ids),
        "critic_flagged_full": sum(bool(initial[case_id]["needs_repair"]) for case_id in safe_ids),
        "critic_flagged_subset": random_budget,
        "random_gate_budget": random_budget,
        "random_gate_seed": seed,
        "random_gate_case_ids_sha256": __import__("hashlib").sha256(
            "".join(f"{case_id}\n" for case_id in sorted(random_ids)).encode()
        ).hexdigest(),
        "gold_used_by_any_selector": False,
        "arms": {
            arm: {
                "path": str(output_dir / ARM_FILES[arm]),
                "records": len(rows),
                "sha256": file_sha256(output_dir / ARM_FILES[arm]),
                "selection_reasons": reason_counts[arm],
            }
            for arm, rows in arms.items()
        },
        "inputs": {
            "safe_sha256": file_sha256(safe_path),
            "initial_sha256": file_sha256(initial_path),
            "generic_sha256": file_sha256(generic_path),
            "self_refine_sha256": file_sha256(self_refine_path),
        },
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe", dest="safe_path", type=Path, required=True)
    parser.add_argument("--initial", dest="initial_path", type=Path, required=True)
    parser.add_argument("--generic", dest="generic_path", type=Path, required=True)
    parser.add_argument("--self-refine", dest="self_refine_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
