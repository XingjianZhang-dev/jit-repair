"""Merge critic retries with a frozen, gold-free non-worsening selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .merge_spider_feedback_retries import load


def choose(previous: dict[str, object], repair: dict[str, object]) -> tuple[dict[str, object], str]:
    if not bool(previous["execution_valid"]):
        return repair, "repair_replaces_execution_failure"
    if not bool(repair["execution_valid"]):
        return previous, "retain_valid_previous_over_invalid_repair"
    if int(repair["critic_score"]) < int(previous["critic_score"]):
        return repair, "repair_reduces_gold_free_critic_score"
    return previous, "retain_previous_without_strict_critic_improvement"


def merge(previous_path: Path, repair_path: Path, output_path: Path, audit_output: Path) -> dict[str, object]:
    if output_path.exists() or audit_output.exists():
        raise FileExistsError("refusing to overwrite critic merge output")
    previous_rows = load(previous_path)
    repairs = {str(row["case_id"]): row for row in load(repair_path)}
    expected = {str(row["case_id"]) for row in previous_rows if bool(row["needs_repair"])}
    if set(repairs) != expected:
        raise ValueError("critic repair output does not exactly cover prior flagged cases")
    merged: list[dict[str, object]] = []
    reasons: dict[str, int] = {}
    for previous in previous_rows:
        case_id = str(previous["case_id"])
        if case_id not in repairs:
            selected, reason = previous, "unflagged_previous_immutable"
        else:
            selected, reason = choose(previous, repairs[case_id])
        selected = dict(selected)
        selected["selection_reason"] = reason
        merged.append(selected)
        reasons[reason] = reasons.get(reason, 0) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    payload = {
        "protocol_id": "spider-gold-free-critic-merge-v1",
        "records": len(merged),
        "flagged_for_repair": len(repairs),
        "selection_reasons": reasons,
        "valid_before": sum(bool(row["execution_valid"]) for row in previous_rows),
        "valid_after": sum(bool(row["execution_valid"]) for row in merged),
        "critic_score_before": sum(int(row["critic_score"]) for row in previous_rows),
        "critic_score_after": sum(int(row["critic_score"]) for row in merged),
        "gold_used_by_selector": False,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.previous, args.repairs, args.output, args.audit_output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
