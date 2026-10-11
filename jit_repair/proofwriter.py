"""ProofWriter structured-data adapter and deterministic conflict transformations."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Iterable

from .executor import KnowledgeBase, Literal, Rule, TruthStatus


VARIABLES = {"someone", "something"}
VERB_LEMMA = {
    "chases": "chase",
    "eats": "eat",
    "likes": "like",
    "needs": "need",
    "sees": "see",
    "visits": "visit",
}
ANSWER_STATUS = {
    "True": TruthStatus.TRUE,
    "False": TruthStatus.FALSE,
    "Unknown": TruthStatus.UNKNOWN,
}


def _symbol(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _term(value: str) -> str:
    return f"?{value}" if value in VARIABLES else _symbol(value)


def atom_to_literal(atom: dict[str, str]) -> Literal:
    subject = _term(atom["subject"])
    if atom["relation"] == "is":
        predicate = _symbol(atom["object"])
        arguments = (subject,)
    else:
        predicate = _symbol(atom["relation"])
        arguments = (subject, _term(atom["object"]))
    polarity = atom["polarity"]
    if polarity not in {"+", "-", "~"}:
        raise ValueError(f"unsupported ProofWriter polarity: {polarity!r}")
    return Literal(predicate, arguments, polarity in {"-", "~"})


def row_to_kb(row: dict[str, object], extra_facts=(), extra_rules=()) -> KnowledgeBase:
    facts = [
        (str(atom["id"]), atom_to_literal(atom)) for atom in row["facts"]
    ] + [(str(atom["id"]), atom_to_literal(atom)) for atom in extra_facts]
    rules: list[Rule] = []
    for item in list(row["rules"]) + list(extra_rules):
        rules.append(
            Rule(
                str(item["id"]),
                tuple(atom_to_literal(atom) for atom in item["premises"]),
                atom_to_literal(item["conclusion"]),
            )
        )
    return KnowledgeBase(facts, rules).saturate()


def entity_text(value: str) -> str:
    return value if value[:1].isupper() else f"the {value}"


def atom_text(atom: dict[str, str]) -> str:
    subject = entity_text(atom["subject"])
    negative = atom["polarity"] == "-"
    if atom["relation"] == "is":
        copula = "is not" if negative else "is"
        return f"{subject} {copula} {atom['object']}."
    obj = entity_text(atom["object"])
    if negative:
        verb = VERB_LEMMA.get(atom["relation"])
        if verb is None:
            raise ValueError(f"no negative-form verb mapping for {atom['relation']}")
        return f"{subject} does not {verb} {obj}."
    return f"{subject} {atom['relation']} {obj}."


def fixed_rule_text(premise: dict[str, str], conclusion: dict[str, str]) -> str:
    premise_clause = atom_text(premise).rstrip(".")
    conclusion_clause = atom_text(conclusion).rstrip(".")
    return f"If {premise_clause} then {conclusion_clause}."


def inject_complement_chain(
    row: dict[str, object], question: dict[str, object], chain_depth: int = 1
) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    """Add a chain supporting the polarity absent from a supported query."""

    if chain_depth < 1:
        raise ValueError("chain_depth must be positive")
    query = deepcopy(question["query"])
    answer = question.get("answer")
    if answer == "True":
        query["polarity"] = "+" if query["polarity"] == "-" else "-"
    elif answer != "False":
        raise ValueError("conflict injection requires an originally True or False query")
    subject = str(query["subject"])
    additions: list[str] = []
    facts: list[dict[str, object]] = []
    rules: list[dict[str, object]] = []
    first = {
        "id": "conflict_fact_1",
        "text": "",
        "subject": subject,
        "relation": "is",
        "object": "flagged",
        "polarity": "+",
    }
    first["text"] = atom_text(first)
    facts.append(first)
    additions.append(str(first["text"]))
    previous = first
    for step in range(1, chain_depth + 1):
        if step == chain_depth:
            conclusion = deepcopy(query)
        else:
            conclusion = {
                "subject": subject,
                "relation": "is",
                "object": f"flagged level {step + 1}",
                "polarity": "+",
            }
        premise = {
            key: previous[key] for key in ("subject", "relation", "object", "polarity")
        }
        rule = {
            "id": f"conflict_rule_{step}",
            "text": fixed_rule_text(premise, conclusion),
            "premises": [premise],
            "conclusion": conclusion,
        }
        rules.append(rule)
        additions.append(str(rule["text"]))
        previous = conclusion
    theory = str(row["theory_text"]).strip()
    return " ".join([theory] + additions), facts, rules


def expected_original_status(question: dict[str, object]) -> TruthStatus:
    return ANSWER_STATUS[str(question["answer"])]


def audit_rows(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    counts = {status.value: 0 for status in TruthStatus}
    counts["mismatch"] = 0
    counts["questions"] = 0
    for row in rows:
        kb = row_to_kb(row)
        for question in row["questions"]:
            observed = kb.status(atom_to_literal(question["query"]))
            expected = expected_original_status(question)
            counts["questions"] += 1
            counts[observed.value] += 1
            counts["mismatch"] += int(observed != expected)
    return counts
