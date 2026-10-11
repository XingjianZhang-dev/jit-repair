"""Reconstruct the official Logic-LM/LoGiPT 600-record ProofWriter test set."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .build_phase0 import program_audit
from .proofwriter import atom_to_literal, expected_original_status, row_to_kb


ANSWER = {"A": "True", "B": "False", "C": "Unknown"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(logiclm: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    indexed = {str(row["id"]): row for row in rows}
    records = []
    audits = []
    label_counts: collections.Counter[str] = collections.Counter()
    query_depths: collections.Counter[int] = collections.Counter()
    for index, item in enumerate(logiclm):
        source = str(item["id"])
        if not source.startswith("ProofWriter_") or "_Q" not in source:
            raise ValueError(f"unexpected Logic-LLM identifier: {source}")
        theory_id, question_suffix = source[len("ProofWriter_") :].rsplit("_Q", 1)
        question_id = f"Q{question_suffix}"
        row = indexed[theory_id]
        questions = {str(question["id"]): question for question in row["questions"]}
        question = questions[question_id]
        if str(item["context"]).strip() != str(row["theory_text"]).strip():
            raise ValueError(f"theory text mismatch for {source}")
        if not str(item["question"]).strip().endswith(str(question["text"]).strip()):
            raise ValueError(f"query text mismatch for {source}")
        expected_answer = ANSWER[str(item["answer"])]
        if expected_answer != str(question["answer"]):
            raise ValueError(f"answer mismatch for {source}")
        kb_status = row_to_kb(row).status(atom_to_literal(question["query"]))
        source_status = expected_original_status(question)
        if kb_status != source_status:
            raise ValueError(f"structured executor mismatch for {source}")
        record = {
            "case_id": f"logiclm_pw_{index:04d}",
            "source_id": theory_id,
            "question_id": question_id,
            "theory": row["theory_text"],
            "query": question["text"],
            "label": source_status.value,
            "source_depth": int(row["depth"]),
            "query_depth": int(question["qdep"]),
            "transformation": "none",
        }
        audit = program_audit(row, question)
        audit["case_id"] = record["case_id"]
        records.append(record)
        audits.append(audit)
        label_counts[record["label"]] += 1
        query_depths[record["query_depth"]] += 1
    if label_counts != {"true": 200, "false": 200, "unknown": 200}:
        raise ValueError(f"unexpected label balance: {label_counts}")
    manifest = {
        "protocol_id": "logiclm-proofwriter-d5-qwen3-14b-v1-frozen-20260821",
        "records": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "query_depth_counts": {str(key): value for key, value in sorted(query_depths.items())},
        "exact_theory_matches": len(records),
        "exact_query_answer_matches": len(records),
        "structured_executor_matches": len(records),
    }
    return records, audits, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logiclm-test", type=Path, required=True)
    parser.add_argument("--structured-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    records, audits, manifest = build(
        json.loads(args.logiclm_test.read_text()),
        pq.read_table(args.structured_test).to_pylist(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    args.audit_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in audits))
    manifest.update({
        "logiclm_test_sha256": sha256(args.logiclm_test),
        "structured_test_sha256": sha256(args.structured_test),
        "output_sha256": sha256(args.output),
        "audit_output_sha256": sha256(args.audit_output),
    })
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
