"""Gold-free question--SQL consistency critic for the Spider external task."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .spider_feedback import render_specific_feedback


@dataclass(frozen=True)
class CriticIssue:
    code: str
    severity: int
    evidence: str
    action: str


def _has(sql: str, pattern: str) -> bool:
    return re.search(pattern, sql, re.IGNORECASE) is not None


def _normalized_numbers(text: str) -> set[str]:
    return {value.lstrip("+") for value in re.findall(r"(?<![\w.])[+-]?\d+(?:\.\d+)?", text)}


def lint_question_sql(question: str, sql: str) -> list[dict[str, object]]:
    """Return deterministic issues based only on the question and predicted SQL."""
    q = question.casefold()
    s = sql.casefold()
    issues: list[CriticIssue] = []

    def add(code: str, severity: int, evidence: str, action: str) -> None:
        issues.append(CriticIssue(code, severity, evidence, action))

    if re.search(r"\b(how many|number of|count of)\b", q) and not _has(s, r"\bcount\s*\("):
        add("missing_count", 3, "The question asks for a count.", "Use COUNT(...) with the required filters/grouping.")
    if re.search(r"\b(average|mean)\b", q) and not _has(s, r"\bavg\s*\("):
        add("missing_average", 3, "The question asks for an average.", "Use AVG(...) on the requested value.")
    if re.search(r"\b(sum of|total (?:amount|cost|price|salary|payment|number of))\b", q) and not _has(
        s, r"\b(?:sum|count)\s*\("
    ):
        add("missing_total_aggregate", 3, "The question asks for a total.", "Use SUM(...) or COUNT(...) as implied by the noun.")

    superlative = re.search(
        r"\b(highest|lowest|largest|smallest|maximum|minimum|oldest|youngest|latest|earliest|most|least)\b",
        q,
    )
    if superlative and not (
        _has(s, r"\b(?:max|min)\s*\(")
        or (_has(s, r"\border\s+by\b") and _has(s, r"\blimit\s+1\b"))
    ):
        add(
            "missing_superlative_operator",
            3,
            f"The question contains superlative '{superlative.group(1)}'.",
            "Use MAX/MIN or ORDER BY with LIMIT 1 as semantically appropriate.",
        )

    if re.search(r"\b(for each|per each|per [a-z]|each [a-z])", q) and _has(
        s, r"\b(?:count|sum|avg|min|max)\s*\("
    ) and not _has(s, r"\bgroup\s+by\b"):
        add("missing_group_by", 2, "The question requests an aggregate for each group.", "Add GROUP BY for the requested grouping entity.")
    if re.search(r"\b(sort(?:ed)?|order(?:ed)?|ascending|descending|alphabetical(?:ly)?)\b", q) and not _has(
        s, r"\border\s+by\b"
    ):
        add("missing_order_by", 2, "The question explicitly requests an ordering.", "Add the requested ORDER BY direction.")
    if re.search(r"\b(distinct|different|unique)\b", q) and not (
        _has(s, r"\bselect\s+distinct\b") or _has(s, r"\bgroup\s+by\b")
    ):
        add("missing_distinct", 2, "The question requests unique results.", "Use SELECT DISTINCT or an equivalent grouping.")
    if re.search(r"\b(not|without|except|other than|excluding)\b", q) and not _has(
        s, r"\b(?:not|except)\b|!=|<>"
    ):
        add("missing_negation", 2, "The question contains an exclusion or negation.", "Encode the exclusion with NOT, NOT IN, !=, <>, or EXCEPT.")

    greater = re.search(r"\b(older than|after|more than|greater than|higher than|over)\b", q)
    if greater and not _has(s, r">|\bbetween\b"):
        add("missing_greater_comparison", 2, f"The question says '{greater.group(1)}'.", "Use the appropriate > or >= comparison.")
    lesser = re.search(r"\b(younger than|before|less than|fewer than|lower than|under)\b", q)
    if lesser and not _has(s, r"<|\bbetween\b"):
        add("missing_lesser_comparison", 2, f"The question says '{lesser.group(1)}'.", "Use the appropriate < or <= comparison.")

    missing_numbers = sorted(_normalized_numbers(question) - _normalized_numbers(sql))
    if missing_numbers:
        add(
            "missing_numeric_literal",
            3,
            "Question number(s) absent from SQL: " + ", ".join(missing_numbers),
            "Copy each required numeric constraint or limit into the SQL.",
        )
    if _has(s, r"\bselect\s+\*") and not re.search(r"\b(all (?:information|details|columns)|everything)\b", q):
        add("overbroad_select_star", 1, "The SQL selects every column without an explicit all-details request.", "Select only the columns needed to answer the question.")

    quoted_values = [value.strip() for value in re.findall(r"[\"']([^\"']+)[\"']", question) if value.strip()]
    missing_quoted = [value for value in quoted_values if value.casefold() not in s]
    if missing_quoted:
        add(
            "missing_quoted_literal",
            3,
            "Quoted question value(s) absent from SQL: " + ", ".join(missing_quoted),
            "Use the quoted value(s) in the relevant predicate.",
        )

    return [asdict(issue) for issue in issues]


def critic_score(issues: Iterable[dict[str, object]]) -> int:
    return sum(int(issue["severity"]) for issue in issues)


def render_critic_specific_feedback(
    execution_error: str | None,
    issues: list[dict[str, object]],
    identifiers: Iterable[str],
) -> str:
    if execution_error is not None:
        base = render_specific_feedback(execution_error, identifiers)
    else:
        base = "\n".join(
            [
                "STATUS: executable_but_consistency_review_required",
                "ERROR_CLASS: question_sql_consistency",
                "SQLITE_DIAGNOSTIC: query executed successfully",
            ]
        )
    if issues:
        lines = [base, "CONSISTENCY_ISSUES:"]
        for issue in issues:
            lines.append(
                f"- {issue['code']} (severity {issue['severity']}): {issue['evidence']} ACTION: {issue['action']}"
            )
        return "\n".join(lines)
    return base + "\nCONSISTENCY_ISSUES: none detected; recheck the complete question against the schema."
