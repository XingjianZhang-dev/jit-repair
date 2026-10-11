"""Gold-blind utilities for the BIRD Mini-Dev generic-forward experiment."""

from __future__ import annotations

from .spider_feedback import (
    execute_readonly,
    extract_sql,
    file_sha256,
    locate_database,
    render_generic_feedback,
    schema_snapshot,
)


BIRD_SQL_SYSTEM_PROMPT = """You translate a natural-language question into SQLite.
Use the supplied expert evidence and database schema. Return exactly one read-only
SELECT query. Common table expressions are allowed. Do not return markdown,
comments, prose, multiple statements, PRAGMA commands, or data-changing SQL."""


def render_initial_user(question: str, evidence: str, schema: str) -> str:
    evidence_text = evidence.strip() or "No additional expert evidence is provided."
    return (
        f"DATABASE SCHEMA WITH VALUE EXAMPLES:\n{schema}\n\n"
        f"EXPERT EVIDENCE:\n{evidence_text}\n\n"
        f"QUESTION:\n{question}\n\nSQL:"
    )


def render_repair_user(
    question: str,
    evidence: str,
    schema: str,
    previous_sql: str,
    feedback: str,
) -> str:
    return (
        render_initial_user(question, evidence, schema)
        + "\n\nPREVIOUS SQL (rejected):\n"
        + previous_sql
        + "\n\nGOLD-FREE REVIEW:\n"
        + feedback
        + "\n\nReturn one complete corrected SQLite SELECT query from scratch."
    )


def critic_text(question: str, evidence: str) -> str:
    """Expose only natural-language question/evidence to the deterministic critic."""
    return question if not evidence.strip() else question + "\nExpert evidence: " + evidence
