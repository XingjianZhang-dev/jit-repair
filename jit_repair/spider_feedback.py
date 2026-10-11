"""Shared, gold-blind utilities for the frozen Spider feedback experiment."""

from __future__ import annotations

import difflib
import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SQL_SYSTEM_PROMPT = """You translate a natural-language question into SQLite.
Use only the supplied database schema. Return exactly one read-only SELECT query.
Common table expressions are allowed. Do not return markdown, comments, prose,
multiple statements, PRAGMA commands, or data-changing SQL."""


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_FINAL_SQL = re.compile(r"FINAL_SQL\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_READ_ONLY_START = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_ERROR_IDENTIFIER = re.compile(
    r"(?:no such (?:column|table|function)|ambiguous column name)\s*:\s*([^\n]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExecutionResult:
    valid: bool
    error: str | None
    error_class: str | None
    rows_observed: int
    elapsed_seconds: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_database(database_dir: Path, db_id: str) -> Path:
    path = database_dir / db_id / f"{db_id}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"Spider database not found: {path}")
    return path


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.text_factory = lambda raw: raw.decode(errors="replace")
    return connection


def schema_snapshot(path: Path) -> tuple[str, tuple[str, ...]]:
    """Return deterministic DDL and identifiers without reading table rows."""
    connection = _readonly_connection(path)
    try:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY lower(name), name"
        ).fetchall()
        ddl: list[str] = []
        identifiers: set[str] = set()
        for table, create_sql in rows:
            table = str(table)
            identifiers.add(table)
            if create_sql:
                statement = str(create_sql).strip()
                ddl.append(statement + ("" if statement.endswith(";") else ";"))
            escaped = table.replace('"', '""')
            for column in connection.execute(f'PRAGMA table_info("{escaped}")'):
                name = str(column[1])
                identifiers.add(name)
                identifiers.add(f"{table}.{name}")
        return "\n".join(ddl), tuple(sorted(identifiers, key=lambda item: (item.casefold(), item)))
    finally:
        connection.close()


def render_initial_user(question: str, schema: str) -> str:
    return f"DATABASE SCHEMA:\n{schema}\n\nQUESTION:\n{question}\n\nSQL:"


def render_repair_user(
    question: str,
    schema: str,
    previous_sql: str,
    feedback: str,
) -> str:
    return (
        render_initial_user(question, schema)
        + "\n\nPREVIOUS SQL (rejected):\n"
        + previous_sql
        + "\n\nEXECUTION FEEDBACK:\n"
        + feedback
        + "\n\nReturn one complete corrected SQLite SELECT query from scratch."
    )


def extract_sql(response: str) -> str:
    """Extract one model SQL response without performing semantic correction."""
    text = response.strip()
    fenced = _FENCE.search(text)
    if fenced is not None:
        text = fenced.group(1).strip()
    final = _FINAL_SQL.search(text)
    if final is not None:
        text = final.group(1).strip()
    text = re.sub(r"^\s*(?:SQL|QUERY)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def classify_execution_error(error: str) -> str:
    lowered = error.casefold()
    if "model_output_parse_failure" in lowered or "empty sql" in lowered:
        return "output_parse_failure"
    if "read-only select" in lowered or "multiple sql statements" in lowered:
        return "unsafe_or_multiple_statement"
    if "interrupted" in lowered or "timeout" in lowered:
        return "execution_timeout"
    if "no such column" in lowered:
        return "unknown_column"
    if "no such table" in lowered:
        return "unknown_table"
    if "ambiguous column name" in lowered:
        return "ambiguous_column"
    if "no such function" in lowered:
        return "unknown_function"
    if "syntax error" in lowered or "incomplete input" in lowered:
        return "sql_syntax"
    if "misuse of aggregate" in lowered or "aggregate" in lowered:
        return "aggregate_error"
    return "sqlite_execution_error"


def execute_readonly(path: Path, sql: str, timeout_seconds: float = 10.0) -> ExecutionResult:
    started = time.perf_counter()
    normalized = sql.strip()
    if not normalized:
        error = "model_output_parse_failure: empty SQL"
        return ExecutionResult(False, error, classify_execution_error(error), 0, 0.0)
    if not _READ_ONLY_START.match(normalized):
        error = "validation_error: expected one read-only SELECT or WITH query"
        return ExecutionResult(False, error, classify_execution_error(error), 0, 0.0)
    without_final = normalized[:-1].rstrip() if normalized.endswith(";") else normalized
    if ";" in without_final:
        error = "validation_error: multiple SQL statements are forbidden"
        return ExecutionResult(False, error, classify_execution_error(error), 0, 0.0)

    connection = _readonly_connection(path)
    deadline = time.monotonic() + timeout_seconds
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cursor = connection.execute(without_final)
        rows = cursor.fetchmany(10_001) if cursor.description else []
        elapsed = time.perf_counter() - started
        return ExecutionResult(True, None, None, len(rows), elapsed)
    except sqlite3.Error as exception:
        error = f"{type(exception).__name__}: {exception}"
        elapsed = time.perf_counter() - started
        return ExecutionResult(False, error, classify_execution_error(error), 0, elapsed)
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def _identifier_matches(error: str, identifiers: Iterable[str]) -> list[str]:
    matched = _ERROR_IDENTIFIER.search(error)
    if matched is None:
        return []
    raw = matched.group(1).strip().strip('`"[]')
    raw = raw.split("(", 1)[0].strip()
    candidates = list(identifiers)
    close = difflib.get_close_matches(raw, candidates, n=5, cutoff=0.35)
    if "." in raw:
        close.extend(difflib.get_close_matches(raw.rsplit(".", 1)[-1], candidates, n=5, cutoff=0.45))
    return list(dict.fromkeys(close))[:5]


def render_specific_feedback(error: str, identifiers: Iterable[str]) -> str:
    error_class = classify_execution_error(error)
    matches = _identifier_matches(error, identifiers)
    lines = [
        "STATUS: execution_failed",
        f"ERROR_CLASS: {error_class}",
        f"SQLITE_DIAGNOSTIC: {error}",
    ]
    if matches:
        lines.append("CLOSE_SCHEMA_IDENTIFIERS: " + ", ".join(matches))
    actions = {
        "output_parse_failure": "Return only one complete SQLite SELECT query.",
        "unsafe_or_multiple_statement": "Replace the response with one read-only SELECT query.",
        "execution_timeout": "Simplify the query and verify join conditions and subqueries.",
        "unknown_column": "Replace the invalid column with an existing schema identifier and qualify it when needed.",
        "unknown_table": "Replace the invalid table with an existing table from the schema.",
        "ambiguous_column": "Qualify the ambiguous column with its table name or alias.",
        "unknown_function": "Use a SQLite-supported function or expression.",
        "sql_syntax": "Repair the syntax at the location described by SQLite.",
        "aggregate_error": "Repair aggregate, GROUP BY, or HAVING usage.",
        "sqlite_execution_error": "Use the diagnostic to repair the query against the supplied schema.",
    }
    lines.append("ACTION: " + actions[error_class])
    return "\n".join(lines)


def render_generic_feedback() -> str:
    """Non-diagnostic control with the same response contract and a useful checklist."""
    return "\n".join(
        [
            "STATUS: execution_failed",
            "ERROR_CLASS: execution_error",
            "SQLITE_DIAGNOSTIC: unavailable",
            "CHECKS: Review the question, table names, column names, joins, predicates, grouping, ordering, and SQLite syntax.",
            "ACTION: Return one complete corrected read-only SQLite SELECT query.",
        ]
    )
