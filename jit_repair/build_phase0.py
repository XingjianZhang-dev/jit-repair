"""Build a deterministic balanced four-valued Phase-0 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .executor import TruthStatus
from .proofwriter import (
    atom_text,
    atom_to_literal,
    audit_rows,
    expected_original_status,
    inject_complement_chain,
    row_to_kb,
)


def literal_payload(literal) -> dict[str, object]:
    return {
        "predicate": literal.predicate,
        "arguments": list(literal.arguments),
        "negated": literal.negated,
    }


def program_audit(
    row: dict[str, object],
    question: dict[str, object],
    extra_facts=(),
    extra_rules=(),
) -> dict[str, object]:
    facts = list(row["facts"]) + list(extra_facts)
    rules = list(row["rules"]) + list(extra_rules)
    return {
        "facts": [
            {"id": str(item["id"]), **literal_payload(atom_to_literal(item))}
            for item in facts
        ],
        "rules": [
            {
                "id": str(item["id"]),
                "premises": [literal_payload(atom_to_literal(atom)) for atom in item["premises"]],
                "conclusion": literal_payload(atom_to_literal(item["conclusion"])),
            }
            for item in rules
        ],
        "query": literal_payload(atom_to_literal(question["query"])),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chain-depth", type=int, default=1)
    parser.add_argument("--min-supported-depth", type=int, default=2)
    parser.add_argument("--min-unknown-theory-depth", type=int, default=3)
    args = parser.parse_args()

    rows = pq.read_table(args.input).to_pylist()
    audit = audit_rows(rows)
    if audit["mismatch"]:
        raise RuntimeError(f"structured executor disagrees with {audit['mismatch']} source labels")
    rng = np.random.default_rng(args.seed)
    pools: dict[TruthStatus, list[tuple[dict, dict]]] = {
        status: [] for status in (TruthStatus.TRUE, TruthStatus.FALSE, TruthStatus.UNKNOWN)
    }
    supported: list[tuple[dict, dict]] = []
    for row in rows:
        for question in row["questions"]:
            status = expected_original_status(question)
            eligible = (
                int(question["qdep"]) >= args.min_supported_depth
                if status in {TruthStatus.TRUE, TruthStatus.FALSE}
                else int(row["depth"]) >= args.min_unknown_theory_depth
            )
            if eligible:
                pools[status].append((row, question))
            if (
                status in {TruthStatus.TRUE, TruthStatus.FALSE}
                and int(question["qdep"]) >= args.min_supported_depth
            ):
                supported.append((row, question))
    for pool in list(pools.values()) + [supported]:
        rng.shuffle(pool)

    entries: list[tuple[dict[str, object], dict[str, object]]] = []
    for status, pool in pools.items():
        if len(pool) < args.per_class:
            raise RuntimeError(f"not enough {status.value} cases: {len(pool)}")
        for row, question in pool[: args.per_class]:
            record = {
                    "source_id": row["id"],
                    "question_id": question["id"],
                    "theory": row["theory_text"],
                    "query": question["text"],
                    "label": status.value,
                    "source_depth": row["depth"],
                    "query_depth": question["qdep"],
                    "transformation": "none",
                }
            entries.append(
                (record, program_audit(row, question))
            )
    if len(supported) < args.per_class:
        raise RuntimeError(f"not enough supported cases for both: {len(supported)}")
    for row, question in supported[: args.per_class]:
        theory, facts, rules = inject_complement_chain(row, question, args.chain_depth)
        kb = row_to_kb(row, facts, rules)
        observed = kb.status(atom_to_literal(question["query"]))
        if observed != TruthStatus.BOTH:
            raise RuntimeError(f"conflict transformation failed for {row['id']} {question['id']}")
        record = {
                "source_id": row["id"],
                "question_id": question["id"],
                "theory": theory,
                "query": atom_text(question["query"]),
                "label": "both",
                "source_depth": row["depth"],
                "query_depth": question["qdep"],
                "transformation": f"independent_complement_chain_depth_{args.chain_depth}",
            }
        entries.append(
            (record, program_audit(row, question, facts, rules))
        )
    order = rng.permutation(len(entries))
    entries = [entries[int(index)] for index in order]
    records = [record for record, _ in entries]
    for index, (record, audit_record) in enumerate(entries):
        record["case_id"] = f"phase0_{index:04d}"
        audit_record["case_id"] = record["case_id"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        with args.audit_output.open("w") as handle:
            for _, audit_record in entries:
                handle.write(json.dumps(audit_record, sort_keys=True) + "\n")
    manifest = {
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "source_rows": len(rows),
        "source_audit": audit,
        "seed": args.seed,
        "per_class": args.per_class,
        "chain_depth": args.chain_depth,
        "min_supported_depth": args.min_supported_depth,
        "min_unknown_theory_depth": args.min_unknown_theory_depth,
        "records": len(records),
        "output_sha256": sha256(args.output),
        "model_input_fields": ["theory", "query"],
        "gold_fields_hidden_from_model": ["label", "source_id", "question_id"],
        "audit_output": str(args.audit_output) if args.audit_output is not None else None,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
