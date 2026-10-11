"""Create gold-separated Spider inputs for a frozen feedback experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .spider_feedback import file_sha256


def prepare(
    source: Path,
    safe_output: Path,
    gold_output: Path,
    audit_output: Path,
    split: str,
    sample_size: int | None = None,
    seed: int = 20260822,
) -> dict[str, object]:
    if any(path.exists() for path in (safe_output, gold_output, audit_output)):
        raise FileExistsError("refusing to overwrite a prepared Spider split")
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Spider source must be a nonempty JSON list")
    selected_indices = list(range(len(rows)))
    if sample_size is not None:
        if not 1 <= sample_size <= len(rows):
            raise ValueError("sample size must be within the source record count")
        selected_indices = sorted(random.Random(seed).sample(selected_indices, sample_size))
    safe_output.parent.mkdir(parents=True, exist_ok=True)
    gold_output.parent.mkdir(parents=True, exist_ok=True)
    db_ids: set[str] = set()
    with safe_output.open("x", encoding="utf-8") as safe_handle, gold_output.open(
        "x", encoding="utf-8"
    ) as gold_handle:
        for index in selected_indices:
            row = rows[index]
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("db_id", "question", "query")):
                raise ValueError(f"invalid Spider row at index {index}")
            db_id = row["db_id"]
            db_ids.add(db_id)
            safe = {
                "case_id": f"spider-{split}-{index:04d}",
                "source_index": index,
                "db_id": db_id,
                "question": row["question"],
            }
            safe_handle.write(json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n")
            gold_handle.write(row["query"].replace("\n", " ") + "\t" + db_id + "\n")
    audit = {
        "protocol_id": "spider-gold-separated-split-v1",
        "split": split,
        "source_records": len(rows),
        "records": len(selected_indices),
        "database_count": len(db_ids),
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "safe_output_sha256": file_sha256(safe_output),
        "gold_output_sha256": file_sha256(gold_output),
        "safe_fields": ["case_id", "source_index", "db_id", "question"],
        "gold_fields_visible_to_generation": False,
        "sample_size": sample_size,
        "selection_seed": seed if sample_size is not None else None,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    parser.add_argument("--gold-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.safe_output, args.gold_output, args.audit_output, args.split, args.sample_size, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
