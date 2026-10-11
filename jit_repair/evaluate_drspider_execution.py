"""Evaluate Dr.Spider arms with the pinned official test-suite semantics."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .drspider_oneuse import file_sha256, locate_drspider_database
from .run_spider_feedback import load_jsonl


_RESULT_EQ: Callable[..., bool] | None = None
_REMOVE_DISTINCT: Callable[[str], str] | None = None
_POSTPROCESS: Callable[[str], str] | None = None
_REPLACE_CUR_YEAR: Callable[[str], str] | None = None


def _init_worker(evaluator_root: str) -> None:
    global _RESULT_EQ, _REMOVE_DISTINCT, _POSTPROCESS, _REPLACE_CUR_YEAR
    sys.path.insert(0, evaluator_root)
    exec_eval = importlib.import_module("exec_eval")
    parser = importlib.import_module("parse")
    _RESULT_EQ = exec_eval.result_eq
    _REMOVE_DISTINCT = parser.remove_distinct
    _POSTPROCESS = exec_eval.postprocess
    _REPLACE_CUR_YEAR = exec_eval.replace_cur_year


def _readonly_execute(path: Path, sql: str, timeout_seconds: float) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.text_factory = lambda raw: raw.decode(errors="ignore")
    deadline = time.monotonic() + timeout_seconds
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def _evaluate_one(
    payload: tuple[int, tuple[str, ...], str, str, float]
) -> tuple[int, int, str | None, str | None]:
    index, database_paths, prediction, gold, timeout_seconds = payload
    assert _RESULT_EQ is not None
    assert _REMOVE_DISTINCT is not None
    assert _POSTPROCESS is not None
    assert _REPLACE_CUR_YEAR is not None
    try:
        gold = _REMOVE_DISTINCT(_POSTPROCESS(gold))
        gold = _REPLACE_CUR_YEAR(gold)
    except Exception as exception:
        return index, 0, None, f"gold preprocessing failed: {type(exception).__name__}: {exception}"
    if not prediction.strip():
        return index, 0, "model_output_parse_failure: empty SQL", None
    try:
        prediction = _REMOVE_DISTINCT(_POSTPROCESS(prediction))
        prediction = _REPLACE_CUR_YEAR(prediction)
    except Exception as exception:
        return index, 0, f"prediction preprocessing failed: {type(exception).__name__}: {exception}", None
    order_matters = "order by" in gold.casefold()
    for database_path in database_paths:
        database = Path(database_path)
        try:
            gold_rows = _readonly_execute(database, gold, timeout_seconds)
        except Exception as exception:
            return index, 0, None, f"gold execution failed: {type(exception).__name__}: {exception}"
        try:
            predicted_rows = _readonly_execute(database, prediction, timeout_seconds)
        except Exception as exception:
            return index, 0, f"{type(exception).__name__}: {exception}", None
        try:
            # Official result_eq samples only to prune impossible column permutations.
            # A fixed seed makes that semantics bit-reproducible across worker schedules.
            random.seed(f"drspider-{index}-{database.name}")
            if not _RESULT_EQ(gold_rows, predicted_rows, order_matters=order_matters):
                return index, 0, None, None
        except Exception as exception:
            return index, 0, None, f"official result_eq failed: {type(exception).__name__}: {exception}"
    return index, 1, None, None


def _aligned_rows(
    predictions: list[dict[str, Any]],
    safe: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    if len(safe) != 15_269 or len(gold) != 15_269:
        raise ValueError("formal safe and gold Dr.Spider files must each have 15,269 rows")
    by_prediction = {str(row["case_id"]): row for row in predictions}
    if len(by_prediction) != len(predictions):
        raise ValueError("duplicate prediction case identifier")
    subset_ids = {str(row["case_id"]) for row in safe if bool(row["in_ablation_subset"])}
    safe_ids = {str(row["case_id"]) for row in safe}
    if set(by_prediction) not in (safe_ids, subset_ids):
        raise ValueError("predictions must exactly cover either the full benchmark or frozen subset")
    aligned: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for safe_row, gold_row in zip(safe, gold, strict=True):
        case_id = str(safe_row["case_id"])
        if str(gold_row["case_id"]) != case_id:
            raise ValueError(f"safe/gold case mismatch at {case_id}")
        for field in ("source_index", "set_index", "set_name", "axis", "q_id_spider_dev", "db_id"):
            if gold_row[field] != safe_row[field]:
                raise ValueError(f"safe/gold {field} mismatch at {case_id}")
        if case_id in by_prediction:
            prediction = by_prediction[case_id]
            if int(prediction["source_index"]) != int(safe_row["source_index"]):
                raise ValueError(f"prediction source index mismatch at {case_id}")
            aligned.append((safe_row, gold_row, prediction))
    if len(aligned) != len(predictions):
        raise ValueError("prediction alignment lost records")
    return aligned


def evaluate(
    predictions_path: Path,
    safe_path: Path,
    gold_path: Path,
    data_root: Path,
    evaluator_root: Path,
    gold_validation_path: Path,
    case_output: Path,
    summary_output: Path,
    workers: int = 16,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    if case_output.exists() or summary_output.exists():
        raise FileExistsError("refusing to overwrite Dr.Spider evaluation output")
    predictions, safe, gold = (
        load_jsonl(path) for path in (predictions_path, safe_path, gold_path)
    )
    aligned = _aligned_rows(predictions, safe, gold)
    validation = json.loads(gold_validation_path.read_text())
    if int(validation["records"]) != 15_269:
        raise ValueError("gold validation record count differs from frozen benchmark")
    fixed_incorrect = {str(row["case_id"]) for row in validation["failures"]}
    safe_ids = {str(row["case_id"]) for row in safe}
    if not fixed_incorrect.issubset(safe_ids):
        raise ValueError("gold validation failure identifier is absent from safe benchmark")

    results: list[tuple[int, int, str | None, str | None] | None] = [None] * len(aligned)
    tasks: list[tuple[int, tuple[str, ...], str, str, float]] = []
    for index, (safe_row, gold_row, prediction) in enumerate(aligned):
        case_id = str(safe_row["case_id"])
        if case_id in fixed_incorrect:
            results[index] = (index, 0, None, None)
            continue
        database = locate_drspider_database(
            data_root, str(safe_row["set_name"]), str(safe_row["db_id"])
        )
        suite = tuple(str(path) for path in sorted(database.parent.glob("*.sqlite")))
        if not suite:
            raise FileNotFoundError(f"empty SQLite test-suite directory for {case_id}")
        tasks.append((index, suite, str(prediction["sql"]), str(gold_row["sql"]), timeout_seconds))
    if tasks:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(evaluator_root),),
        ) as executor:
            for completed, result in enumerate(executor.map(_evaluate_one, tasks), 1):
                results[result[0]] = result
                if completed == len(tasks) or completed % 500 == 0:
                    print(f"Dr.Spider official execution {completed}/{len(tasks)}", flush=True)

    cases: list[dict[str, Any]] = []
    for (safe_row, _, prediction), result in zip(aligned, results, strict=True):
        assert result is not None
        _, correct, prediction_error, evaluator_error = result
        case_id = str(safe_row["case_id"])
        cases.append(
            {
                "case_id": case_id,
                "source_index": safe_row["source_index"],
                "set_index": safe_row["set_index"],
                "set_name": safe_row["set_name"],
                "axis": safe_row["axis"],
                "q_id_spider_dev": safe_row["q_id_spider_dev"],
                "db_id": safe_row["db_id"],
                "correct": bool(correct),
                "prediction": "correct" if correct else "incorrect",
                "gold": "correct",
                "execution_valid": bool(prediction["execution_valid"]),
                "prediction_execution_error": prediction_error,
                "official_evaluator_error": evaluator_error,
                "gold_self_inconsistent_fixed_incorrect": case_id in fixed_incorrect,
            }
        )
    case_output.parent.mkdir(parents=True, exist_ok=True)
    with case_output.open("x", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def grouped(field: str) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cases:
            groups[str(row[field])].append(row)
        return {
            group: {
                "records": len(rows),
                "correct": sum(bool(row["correct"]) for row in rows),
                "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            }
            for group, rows in sorted(groups.items())
        }

    summary: dict[str, Any] = {
        "protocol_id": "drspider-pinned-official-test-suite-execution-v1",
        "evaluation_database_mode": "drspider_post_perturbation_official_test_suite",
        "official_result_comparison": "result_eq_with_gold_order_by_semantics",
        "official_plug_value": False,
        "official_keep_distinct": False,
        "records": len(cases),
        "execution_accuracy": sum(bool(row["correct"]) for row in cases) / len(cases),
        "execution_validity": sum(bool(row["execution_valid"]) for row in cases) / len(cases),
        "by_axis": grouped("axis"),
        "by_set": grouped("set_name"),
        "official_evaluator_errors": sum(row["official_evaluator_error"] is not None for row in cases),
        "prediction_execution_errors": sum(row["prediction_execution_error"] is not None for row in cases),
        "gold_self_inconsistent_fixed_incorrect": sum(
            bool(row["gold_self_inconsistent_fixed_incorrect"]) for row in cases
        ),
        "workers": workers,
        "timeout_seconds_per_query": timeout_seconds,
        "official_evaluator_commit": "e97acc546ecbee8fa27fa8dbf025ef61493a876c",
        "predictions_sha256": file_sha256(predictions_path),
        "safe_sha256": file_sha256(safe_path),
        "gold_sha256": file_sha256(gold_path),
        "gold_validation_sha256": file_sha256(gold_validation_path),
        "official_evaluation_py_sha256": file_sha256(evaluator_root / "evaluation.py"),
        "official_exec_eval_py_sha256": file_sha256(evaluator_root / "exec_eval.py"),
        "official_parse_py_sha256": file_sha256(evaluator_root / "parse.py"),
        "case_output_sha256": file_sha256(case_output),
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", dest="predictions_path", type=Path, required=True)
    parser.add_argument("--safe", dest="safe_path", type=Path, required=True)
    parser.add_argument("--gold", dest="gold_path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--gold-validation", dest="gold_validation_path", type=Path, required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
