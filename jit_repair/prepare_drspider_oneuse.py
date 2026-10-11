"""Create full post-perturbation and nested ablation Dr.Spider gold-isolated assets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .drspider_oneuse import (
    PERTURBATION_SETS,
    SAFE_FIELDS,
    axis_for_set,
    deterministic_rank,
    file_sha256,
    locate_drspider_database,
    schema_snapshot,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_gold(line: str) -> tuple[str, str]:
    fields = line.rstrip("\n").rsplit("\t", 1)
    if len(fields) != 2 or not all(field.strip() for field in fields):
        raise ValueError("Dr.Spider gold line does not contain SQL and db_id")
    return fields[0].strip(), fields[1].strip()


def _subset_indices(
    set_name: str,
    rows: list[dict[str, Any]],
    seed: int,
    per_set_cap: int,
) -> set[int]:
    by_qid: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_qid[int(row["q_id_spider_dev"])].append(index)
    representatives: list[tuple[str, int, int]] = []
    for qid, indices in by_qid.items():
        chosen = min(indices, key=lambda index: deterministic_rank(seed, set_name, qid, index, "variant"))
        representatives.append((deterministic_rank(seed, set_name, qid, "qid"), qid, chosen))
    representatives.sort()
    return {index for _, _, index in representatives[:per_set_cap]}


def prepare(
    data_root: Path,
    safe_output: Path,
    gold_output: Path,
    audit_output: Path,
    database_hash_output: Path,
    seed: int = 20260822,
    per_set_cap: int = 100,
) -> dict[str, Any]:
    for output in (safe_output, gold_output, audit_output, database_hash_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    if per_set_cap <= 0:
        raise ValueError("per_set_cap must be positive")

    safe: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    set_counts: dict[str, dict[str, int]] = {}
    schema_cache: dict[tuple[str, str], str] = {}
    databases: dict[str, Path] = {}
    global_index = 0

    for set_name in PERTURBATION_SETS:
        directory = data_root / set_name
        questions_path = directory / "questions_post_perturbation.json"
        gold_path = directory / "gold_post_perturbation.sql"
        questions = json.loads(questions_path.read_text(encoding="utf-8"))
        gold_lines = gold_path.read_text(encoding="utf-8").splitlines()
        if len(questions) != len(gold_lines) or not questions:
            raise ValueError(f"question/gold alignment failed for {set_name}")
        chosen = _subset_indices(set_name, questions, seed, per_set_cap)
        axis = axis_for_set(set_name)
        for set_index, (question_row, gold_line) in enumerate(zip(questions, gold_lines, strict=True)):
            required = {"db_id", "q_id_spider_dev", "question", "query"}
            if not required.issubset(question_row):
                raise ValueError(f"source fields missing for {set_name} row {set_index}")
            gold_sql, gold_db_id = _parse_gold(gold_line)
            db_id = str(question_row["db_id"])
            if gold_db_id != db_id or gold_sql != str(question_row["query"]).strip():
                raise ValueError(f"post question/gold mismatch for {set_name} row {set_index}")
            database = locate_drspider_database(data_root, set_name, db_id)
            database_key = str(database.resolve())
            databases[database_key] = database
            cache_key = (set_name if axis == "DB" else "Spider-dev", db_id)
            if cache_key not in schema_cache:
                schema_cache[cache_key] = schema_snapshot(database)[0]
            case_id = f"drspider-{set_name}-{set_index:05d}"
            safe.append(
                {
                    "case_id": case_id,
                    "source_index": global_index,
                    "set_index": set_index,
                    "set_name": set_name,
                    "axis": axis,
                    "q_id_spider_dev": int(question_row["q_id_spider_dev"]),
                    "db_id": db_id,
                    "question": str(question_row["question"]).strip(),
                    "schema": schema_cache[cache_key],
                    "in_ablation_subset": set_index in chosen,
                }
            )
            gold.append(
                {
                    "case_id": case_id,
                    "source_index": global_index,
                    "set_index": set_index,
                    "set_name": set_name,
                    "axis": axis,
                    "q_id_spider_dev": int(question_row["q_id_spider_dev"]),
                    "db_id": db_id,
                    "sql": gold_sql,
                }
            )
            global_index += 1
        set_counts[set_name] = {
            "records": len(questions),
            "unique_qids": len({int(row["q_id_spider_dev"]) for row in questions}),
            "ablation_subset_records": len(chosen),
        }

    if len(safe) != 15_269 or len(gold) != 15_269:
        raise ValueError(f"expected 15,269 Dr.Spider rows, observed {len(safe)}")
    if len({row["case_id"] for row in safe}) != len(safe):
        raise ValueError("duplicate Dr.Spider case identifier")
    if any(set(row) != SAFE_FIELDS for row in safe):
        raise AssertionError("safe-field contract changed")
    if any(any(field in row for field in ("query", "sql", "gold")) for row in safe):
        raise AssertionError("gold SQL leaked into safe rows")

    database_hash_output.parent.mkdir(parents=True, exist_ok=True)
    database_hash_output.write_text(
        "".join(f"{file_sha256(path)}  {path.resolve()}\n" for path in sorted(databases.values(), key=lambda item: str(item.resolve()))),
        encoding="utf-8",
    )
    _write_jsonl(safe_output, safe)
    _write_jsonl(gold_output, gold)

    axis_counts = {
        axis: {
            "records": sum(row["axis"] == axis for row in safe),
            "ablation_subset_records": sum(row["axis"] == axis and row["in_ablation_subset"] for row in safe),
        }
        for axis in ("DB", "NLQ", "SQL")
    }
    audit = {
        "protocol_id": "drspider-post-perturbation-oneuse-gold-isolation-audit-v1",
        "data_root": str(data_root),
        "records": len(safe),
        "sets": len(PERTURBATION_SETS),
        "unique_spider_dev_qids": len({row["q_id_spider_dev"] for row in safe}),
        "ablation_subset_records": sum(bool(row["in_ablation_subset"]) for row in safe),
        "ablation_subset_rule": "For every set, choose one deterministic variant per q_id_spider_dev, then the first at most 100 qids by SHA-256 rank.",
        "ablation_subset_uses_gold": False,
        "per_set_cap": per_set_cap,
        "seed": seed,
        "set_counts": set_counts,
        "axis_counts": axis_counts,
        "database_files": len(databases),
        "database_hash_manifest_sha256": file_sha256(database_hash_output),
        "safe_fields": sorted(SAFE_FIELDS),
        "gold_fields_absent_from_safe": True,
        "safe_sha256": file_sha256(safe_output),
        "gold_sha256": file_sha256(gold_output),
        "official_gold_execution_validation": "pending_result_blind_official_test_suite_audit",
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    parser.add_argument("--gold-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--database-hash-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--per-set-cap", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
