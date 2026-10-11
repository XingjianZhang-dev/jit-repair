"""Initial-failure classes of ProofWriter logic programs (validator stage order of jit_repair.run_model.execute_dsl); used by X2."""


def is_correct(r):
    return int(r.get("prediction") is not None and r.get("prediction") == r.get("gold"))


import re as _re
_PARSE = _re.compile(r"^(invalid DSL literal|invalid DSL arguments|invalid DSL rule|unexpected DSL line|DSL query is missing|"
                     r"DSL must contain exactly one query|generic .is. predicate|no JSON|compiled output is not an object)")
_STRUCT = _re.compile(r"^(expected \d+ facts|expected \d+ rules|(fact|rule) identifiers do not match numbered input|"
                      r"fact and rule ids must be globally unique)")
_ALIGN_CLAUSE = _re.compile(r"^(\S+ predicate is not aligned to its source sentence|\S+ constant .* is absent from its source sentence|"
                            r"\S+ must remain a ground rule for named entities|\S+ must retain a variable for its generic entity|"
                            r"\S+ fact cannot contain variables)$", _re.S)
_EXEC = _re.compile(r"^(rule \S+ is not range-restricted|fact \S+ is not ground|knowledge base did not reach a fixpoint|"
                    r"unbound variable in conclusion|negated must be Boolean|literal must have a predicate)")
CATEGORIES = ["correct", "parse", "structural", "alignment", "execution", "executes_but_wrong", "other"]


def error_body(err):
    m = _re.match(r"^([A-Za-z_][\w.]*):\s(.*)$", err, _re.S)
    return (m.group(1), m.group(2)) if m else (None, err)


def classify(rec):
    """correct / parse / structural / alignment / execution / executes_but_wrong / other."""
    err = rec.get("error")
    if err is None:
        return "correct" if is_correct(rec) else "executes_but_wrong"
    exc, body = error_body(err)
    if _PARSE.match(body):
        return "parse"
    if _STRUCT.match(body):
        return "structural"
    clauses = body.split("; ")
    if all(_ALIGN_CLAUSE.match(c) for c in clauses):
        return "alignment"
    if _EXEC.match(body):
        return "execution"
    if exc in ("RuntimeError", "RecursionError"):
        return "execution"
    return "other"
