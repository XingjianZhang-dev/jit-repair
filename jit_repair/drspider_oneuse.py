"""Gold-blind utilities for the Dr.Spider OOD one-use experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .spider_feedback import (
    execute_readonly,
    extract_sql,
    file_sha256,
    render_generic_feedback,
    schema_snapshot,
)


PERTURBATION_SETS = (
    "DB_DBcontent_equivalence",
    "DB_schema_abbreviation",
    "DB_schema_synonym",
    "NLQ_column_attribute",
    "NLQ_column_carrier",
    "NLQ_column_synonym",
    "NLQ_column_value",
    "NLQ_keyword_carrier",
    "NLQ_keyword_synonym",
    "NLQ_multitype",
    "NLQ_others",
    "NLQ_value_synonym",
    "SQL_DB_number",
    "SQL_DB_text",
    "SQL_NonDB_number",
    "SQL_comparison",
    "SQL_sort_order",
)

DRSPIDER_SQL_SYSTEM_PROMPT = """You translate a natural-language question into SQLite.
Use only the supplied database schema. Return exactly one read-only SELECT query.
Common table expressions are allowed. Do not return markdown, comments, prose,
multiple statements, PRAGMA commands, or data-changing SQL."""

SELF_REFINE_FEEDBACK_SYSTEM_PROMPT = """You are a meticulous SQLite reviewer.
Given a database schema, a question, and a proposed SQL query, identify concrete
semantic or SQLite errors. Check tables, columns, joins, predicates, aggregation,
grouping, ordering, limits, and whether the query answers every part of the
question. If no error is apparent, say exactly: The SQL is correct. Return only
concise feedback; do not return a replacement SQL query."""


def axis_for_set(set_name: str) -> str:
    axis = set_name.split("_", 1)[0]
    if axis not in {"DB", "NLQ", "SQL"}:
        raise ValueError(f"unknown Dr.Spider axis for {set_name}")
    return axis


def locate_drspider_database(data_root: Path, set_name: str, db_id: str) -> Path:
    if axis_for_set(set_name) == "DB":
        path = data_root / set_name / "database_post_perturbation" / db_id / f"{db_id}.sqlite"
    else:
        path = data_root / "Spider-dev" / "databases" / db_id / f"{db_id}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"Dr.Spider database not found: {path}")
    return path


def render_initial_user(question: str, schema: str) -> str:
    return f"DATABASE SCHEMA:\n{schema}\n\nQUESTION:\n{question}\n\nSQL:"


def render_generic_repair_user(question: str, schema: str, previous_sql: str) -> str:
    return (
        render_initial_user(question, schema)
        + "\n\nPREVIOUS SQL (rejected):\n"
        + previous_sql
        + "\n\nGOLD-FREE REVIEW:\n"
        + render_generic_feedback()
        + "\n\nReturn one complete corrected SQLite SELECT query from scratch."
    )


def render_self_refine_feedback_user(question: str, schema: str, previous_sql: str) -> str:
    return (
        f"DATABASE SCHEMA:\n{schema}\n\nQUESTION:\n{question}\n\n"
        f"PROPOSED SQL:\n{previous_sql}\n\nFEEDBACK:"
    )


def render_self_refine_iterate_user(
    question: str,
    schema: str,
    previous_sql: str,
    feedback: str,
) -> str:
    return (
        render_initial_user(question, schema)
        + "\n\nPREVIOUS SQL:\n"
        + previous_sql
        + "\n\nSELF-FEEDBACK:\n"
        + feedback
        + "\n\nReturn one complete refined SQLite SELECT query from scratch."
    )


def deterministic_rank(seed: int, *parts: object) -> str:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


SAFE_FIELDS = {
    "case_id",
    "source_index",
    "set_index",
    "set_name",
    "axis",
    "q_id_spider_dev",
    "db_id",
    "question",
    "schema",
    "in_ablation_subset",
}


__all__ = [
    "DRSPIDER_SQL_SYSTEM_PROMPT",
    "PERTURBATION_SETS",
    "SAFE_FIELDS",
    "SELF_REFINE_FEEDBACK_SYSTEM_PROMPT",
    "axis_for_set",
    "deterministic_rank",
    "execute_readonly",
    "extract_sql",
    "file_sha256",
    "locate_drspider_database",
    "render_generic_repair_user",
    "render_initial_user",
    "render_self_refine_feedback_user",
    "render_self_refine_iterate_user",
    "schema_snapshot",
]
