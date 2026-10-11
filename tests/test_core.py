"""Standard-library tests for the deterministic method core (executor, SQL checker, critic, selector)."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT))

from jit_repair import KnowledgeBase, Literal, TruthStatus  # noqa: E402
from jit_repair.merge_spider_critic_retries import choose  # noqa: E402
from jit_repair.spider_critic_feedback import (  # noqa: E402
    critic_score,
    lint_question_sql,
)
from jit_repair.spider_feedback import execute_readonly, extract_sql  # noqa: E402


class CoreMethodTests(unittest.TestCase):
    def test_paraconsistent_executor_preserves_conflict_without_explosion(self) -> None:
        positive = Literal("verified", ("item",))
        negative = positive.complement()
        unrelated = Literal("approved", ("item",))
        knowledge = KnowledgeBase(
            [("fact_positive", positive), ("fact_negative", negative)], []
        ).saturate()

        self.assertEqual(knowledge.status(positive), TruthStatus.BOTH)
        self.assertEqual(knowledge.status(unrelated), TruthStatus.UNKNOWN)

    def test_sql_extraction_and_read_only_execution_contract(self) -> None:
        self.assertEqual(
            extract_sql("```sql\nSELECT COUNT(*) FROM records;\n```"),
            "SELECT COUNT(*) FROM records",
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "minimal.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE records(value INTEGER)")
                connection.executemany(
                    "INSERT INTO records(value) VALUES (?)", [(1,), (2,), (3,)]
                )
                connection.commit()
            finally:
                connection.close()

            valid = execute_readonly(database, "SELECT COUNT(*) FROM records")
            unsafe = execute_readonly(database, "DELETE FROM records")
            multiple = execute_readonly(
                database, "SELECT COUNT(*) FROM records; SELECT 1"
            )

        self.assertTrue(valid.valid)
        self.assertFalse(unsafe.valid)
        self.assertEqual(unsafe.error_class, "unsafe_or_multiple_statement")
        self.assertFalse(multiple.valid)
        self.assertEqual(multiple.error_class, "unsafe_or_multiple_statement")

    def test_gold_free_critic_detects_question_sql_mismatch(self) -> None:
        issues = lint_question_sql(
            "How many records have a value over 2?",
            "SELECT value FROM records",
        )
        codes = {str(issue["code"]) for issue in issues}
        self.assertIn("missing_count", codes)
        self.assertIn("missing_greater_comparison", codes)
        self.assertIn("missing_numeric_literal", codes)
        self.assertGreater(critic_score(issues), 0)

    def test_selector_rejects_invalid_or_non_improving_repairs(self) -> None:
        previous = {"execution_valid": True, "critic_score": 2, "sql": "initial"}
        invalid = {"execution_valid": False, "critic_score": 0, "sql": "invalid"}
        tied = {"execution_valid": True, "critic_score": 2, "sql": "tied"}

        selected, reason = choose(previous, invalid)
        self.assertIs(selected, previous)
        self.assertEqual(reason, "retain_valid_previous_over_invalid_repair")

        selected, reason = choose(previous, tied)
        self.assertIs(selected, previous)
        self.assertEqual(reason, "retain_previous_without_strict_critic_improvement")

    def test_selector_accepts_valid_strict_improvement(self) -> None:
        previous = {"execution_valid": True, "critic_score": 3, "sql": "initial"}
        repair = {"execution_valid": True, "critic_score": 1, "sql": "repair"}

        selected, reason = choose(previous, repair)
        self.assertIs(selected, repair)
        self.assertEqual(reason, "repair_reduces_gold_free_critic_score")


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoreMethodTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("PASS: deterministic core implementation tests completed.")


if __name__ == "__main__":
    main()
