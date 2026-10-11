"""Split official BIRD Mini-Dev into gold-free generation and isolated gold files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bird_minidev_feedback import execute_readonly, file_sha256, locate_database


SOURCE_FIELDS = {
    "question_id",
    "db_id",
    "question",
    "evidence",
    "full_question",
    "SQL",
    "schema",
    "prompt",
    "difficulty",
}
SAFE_FIELDS = {
    "case_id",
    "source_index",
    "question_id",
    "db_id",
    "question",
    "evidence",
    "schema",
    "difficulty",
}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(
    source: Path,
    database_root: Path,
    safe_output: Path,
    gold_output: Path,
    audit_output: Path,
    database_hash_output: Path,
    known_gold_timeout_case_ids: tuple[str, ...] = (),
    validation_timeout_seconds: float = 60.0,
) -> dict[str, object]:
    for path in (safe_output, gold_output, audit_output, database_hash_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 500:
        raise ValueError(f"expected exactly 500 Mini-Dev rows, observed {len(rows)}")
    if any(set(row) != SOURCE_FIELDS for row in rows):
        raise ValueError("official source fields differ from the frozen contract")
    question_ids = [str(row["question_id"]) for row in rows]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("duplicate question_id")
    db_ids = sorted({str(row["db_id"]) for row in rows})
    if len(db_ids) != 11:
        raise ValueError(f"expected 11 Mini-Dev databases, observed {len(db_ids)}")

    database_hashes: list[tuple[str, str]] = []
    for db_id in db_ids:
        database = locate_database(database_root, db_id)
        database_hashes.append((file_sha256(database), str(database)))
    database_hash_output.parent.mkdir(parents=True, exist_ok=True)
    database_hash_output.write_text(
        "".join(f"{digest}  {path}\n" for digest, path in database_hashes), encoding="utf-8"
    )

    safe: list[dict[str, object]] = []
    gold: list[dict[str, object]] = []
    invalid_gold: list[dict[str, object]] = []
    timed_out_gold: list[str] = []
    allowed_timeouts = set(known_gold_timeout_case_ids)
    for index, row in enumerate(rows):
        if row["difficulty"] not in {"simple", "moderate", "challenging"}:
            raise ValueError(f"unexpected difficulty at row {index}")
        for field in ("question_id", "db_id", "question", "schema", "SQL"):
            if not str(row[field]).strip():
                raise ValueError(f"empty {field} at row {index}")
        case_id = f"bird-mini-dev-{row['question_id']}"
        safe.append({
            "case_id": case_id,
            "source_index": index,
            "question_id": row["question_id"],
            "db_id": row["db_id"],
            "question": row["question"],
            "evidence": row["evidence"],
            "schema": row["schema"],
            "difficulty": row["difficulty"],
        })
        gold_sql = str(row["SQL"]).strip()
        result = execute_readonly(
            locate_database(database_root, str(row["db_id"])), gold_sql, validation_timeout_seconds
        )
        if not result.valid:
            if result.error_class == "execution_timeout" and case_id in allowed_timeouts:
                timed_out_gold.append(case_id)
            else:
                invalid_gold.append({"case_id": case_id, "error": result.error})
        gold.append({
            "case_id": case_id,
            "question_id": row["question_id"],
            "db_id": row["db_id"],
            "sql": gold_sql,
        })
    if invalid_gold:
        raise ValueError(f"official gold SQL execution failures: {invalid_gold[:5]}")
    if set(timed_out_gold) != allowed_timeouts:
        raise ValueError(
            f"known gold timeout set differs: expected={sorted(allowed_timeouts)}, "
            f"observed={sorted(timed_out_gold)}"
        )
    if any(set(row) != SAFE_FIELDS for row in safe):
        raise AssertionError("safe-field contract changed")

    safe_output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(safe_output, safe)
    _write_jsonl(gold_output, gold)
    audit = {
        "protocol_id": "bird-minidev-gold-isolation-audit-v1",
        "source": str(source),
        "source_sha256": file_sha256(source),
        "records": len(rows),
        "unique_question_ids": len(set(question_ids)),
        "databases": len(db_ids),
        "database_ids": db_ids,
        "official_gold_sql_execution_valid": len(rows) - len(timed_out_gold),
        "known_gold_timeout_cases_retained_as_incorrect": sorted(timed_out_gold),
        "known_gold_timeout_count": len(timed_out_gold),
        "unexpected_gold_execution_failures": 0,
        "validation_timeout_seconds": validation_timeout_seconds,
        "safe_fields": sorted(SAFE_FIELDS),
        "gold_fields_absent_from_safe": all("SQL" not in row and "sql" not in row for row in safe),
        "prompt_and_full_question_absent_from_safe": all(
            "prompt" not in row and "full_question" not in row for row in safe
        ),
        "safe_sha256": file_sha256(safe_output),
        "gold_sha256": file_sha256(gold_output),
        "database_hash_manifest_sha256": file_sha256(database_hash_output),
    }
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    parser.add_argument("--gold-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--database-hash-output", type=Path, required=True)
    parser.add_argument("--known-gold-timeout-case-id", action="append", default=[])
    parser.add_argument("--validation-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    values = vars(args)
    values["known_gold_timeout_case_ids"] = tuple(values.pop("known_gold_timeout_case_id"))
    print(json.dumps(prepare(**values), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
