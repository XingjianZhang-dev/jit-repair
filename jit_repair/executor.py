"""A small range-restricted paraconsistent Datalog executor with provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


class TruthStatus(str, Enum):
    TRUE = "true"
    FALSE = "false"
    BOTH = "both"
    UNKNOWN = "unknown"


@dataclass(frozen=True, order=True)
class Literal:
    predicate: str
    arguments: tuple[str, ...]
    negated: bool = False

    def complement(self) -> "Literal":
        return Literal(self.predicate, self.arguments, not self.negated)

    def ground(self, substitution: Mapping[str, str]) -> "Literal":
        arguments = tuple(substitution.get(term, term) for term in self.arguments)
        if any(term.startswith("?") for term in arguments):
            raise ValueError(f"unbound variable in conclusion: {self}")
        return Literal(self.predicate, arguments, self.negated)


@dataclass(frozen=True)
class Rule:
    identifier: str
    premises: tuple[Literal, ...]
    conclusion: Literal

    def __post_init__(self) -> None:
        premise_variables = {
            term
            for literal in self.premises
            for term in literal.arguments
            if term.startswith("?")
        }
        conclusion_variables = {
            term for term in self.conclusion.arguments if term.startswith("?")
        }
        if not conclusion_variables <= premise_variables:
            missing = sorted(conclusion_variables - premise_variables)
            raise ValueError(f"rule {self.identifier} is not range-restricted: {missing}")


@dataclass(frozen=True)
class Proof:
    literal: Literal
    source: str
    premises: tuple["Proof", ...] = ()

    @property
    def depth(self) -> int:
        return 0 if not self.premises else 1 + max(proof.depth for proof in self.premises)

    @property
    def size(self) -> int:
        return 1 + sum(proof.size for proof in self.premises)

    def sources(self) -> tuple[str, ...]:
        output = [self.source]
        for proof in self.premises:
            output.extend(proof.sources())
        return tuple(output)


def _match(pattern: Literal, fact: Literal, initial: Mapping[str, str]) -> dict[str, str] | None:
    if (
        pattern.predicate != fact.predicate
        or pattern.negated != fact.negated
        or len(pattern.arguments) != len(fact.arguments)
    ):
        return None
    substitution = dict(initial)
    for term, value in zip(pattern.arguments, fact.arguments, strict=True):
        if term.startswith("?"):
            previous = substitution.get(term)
            if previous is not None and previous != value:
                return None
            substitution[term] = value
        elif term != value:
            return None
    return substitution


class KnowledgeBase:
    """Forward-chain explicit negation without deriving arbitrary explosions."""

    def __init__(self, facts: Iterable[tuple[str, Literal]], rules: Iterable[Rule]) -> None:
        self.rules = tuple(rules)
        self.proofs: dict[Literal, Proof] = {}
        for identifier, literal in facts:
            if any(term.startswith("?") for term in literal.arguments):
                raise ValueError(f"fact {identifier} is not ground")
            proof = Proof(literal=literal, source=identifier)
            self._keep_shorter(proof)

    def _keep_shorter(self, proof: Proof) -> bool:
        existing = self.proofs.get(proof.literal)
        if existing is None or (proof.depth, proof.size, proof.source) < (
            existing.depth,
            existing.size,
            existing.source,
        ):
            self.proofs[proof.literal] = proof
            return True
        return False

    def _premise_matches(
        self,
        premises: tuple[Literal, ...],
        index: int = 0,
        substitution: Mapping[str, str] | None = None,
        selected: tuple[Proof, ...] = (),
    ) -> Iterable[tuple[dict[str, str], tuple[Proof, ...]]]:
        substitution = {} if substitution is None else substitution
        if index == len(premises):
            yield dict(substitution), selected
            return
        pattern = premises[index]
        for fact, proof in tuple(self.proofs.items()):
            matched = _match(pattern, fact, substitution)
            if matched is not None:
                yield from self._premise_matches(
                    premises, index + 1, matched, selected + (proof,)
                )

    def saturate(self, max_rounds: int = 1_000) -> "KnowledgeBase":
        for _ in range(max_rounds):
            changed = False
            for rule in self.rules:
                for substitution, premise_proofs in self._premise_matches(rule.premises):
                    conclusion = rule.conclusion.ground(substitution)
                    changed |= self._keep_shorter(
                        Proof(conclusion, rule.identifier, premise_proofs)
                    )
            if not changed:
                return self
        raise RuntimeError(f"knowledge base did not reach a fixpoint in {max_rounds} rounds")

    def status(self, query: Literal) -> TruthStatus:
        positive = query in self.proofs
        negative = query.complement() in self.proofs
        if positive and negative:
            return TruthStatus.BOTH
        if positive:
            return TruthStatus.TRUE
        if negative:
            return TruthStatus.FALSE
        return TruthStatus.UNKNOWN

    def query(self, query: Literal) -> dict[str, object]:
        return {
            "status": self.status(query).value,
            "positive_proof": self.proofs.get(query),
            "negative_proof": self.proofs.get(query.complement()),
        }
