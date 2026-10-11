"""Merge a sparse Spider repair attempt into the previous full-stage results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def merge(previous_path: Path, repair_path: Path, output_path: Path) -> dict[str, int]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    previous = load(previous_path)
    repairs = {str(row["case_id"]): row for row in load(repair_path)}
    expected_repairs = {str(row["case_id"]) for row in previous if not bool(row["execution_valid"])}
    if set(repairs) != expected_repairs:
        raise ValueError("repair output does not exactly cover prior execution failures")
    merged = [repairs.get(str(row["case_id"]), row) for row in previous]
    if len({str(row["case_id"]) for row in merged}) != len(merged):
        raise ValueError("duplicate case identifier in merged output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "records": len(merged),
        "repaired": len(repairs),
        "valid_before": sum(bool(row["execution_valid"]) for row in previous),
        "valid_after": sum(bool(row["execution_valid"]) for row in merged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.previous, args.repairs, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
