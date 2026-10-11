"""Run direct, CoT, or compile-execute inference with a local chat model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from .executor import KnowledgeBase, Literal, Proof, Rule


LABELS = ("true", "false", "both", "unknown")
RELATION_CANONICAL = {
    "chase": "chases",
    "eat": "eats",
    "like": "likes",
    "need": "needs",
    "see": "sees",
    "visit": "visits",
}

SEMANTICS = """Use only the supplied theory. Use forward rules exactly as written; do not use contraposition or outside knowledge. Explicit negative statements are independent evidence. Classify the query using four values:
- true: the query is supported and its negation is not supported;
- false: the negation is supported and the query is not supported;
- both: both the query and its negation are supported;
- unknown: neither is supported."""

DIRECT_SYSTEM = SEMANTICS + """
Do not explain, derive, restate the theory, or output any other text. Your entire response must be exactly one of these four lines:
FINAL: true
FINAL: false
FINAL: both
FINAL: unknown"""
COT_SYSTEM = SEMANTICS + "\nShow a concise derivation, then return exactly one final line: FINAL: true|false|both|unknown."
COMPILE_SYSTEM = """Translate the supplied natural-language theory and query into JSON only. Do not solve the query and do not add facts. Use this exact schema:
{"facts":[{"id":"f1","predicate":"blue","arguments":["charlie"],"negated":false}],"rules":[{"id":"r1","premises":[{"predicate":"blue","arguments":["?x"],"negated":false}],"conclusion":{"predicate":"kind","arguments":["?x"],"negated":false}}],"query":{"predicate":"kind","arguments":["charlie"],"negated":false}}
For properties, use the adjective as predicate and one subject argument. For relations, use the verb as predicate and subject/object arguments. Lowercase snake_case all constants and predicates. In a rule, someone/something/they/it referring to the same entity must use the same variable such as ?x. Preserve every explicit negation with negated=true. Translate every sentence exactly once. Output one valid JSON object and nothing else."""
DSL_SYSTEM = """Compile each numbered source sentence into exactly one matching compact DSL line. Do not solve the query, omit or add knowledge, or explain. The input already marks facts F1... and rules R1.... Copy every identifier exactly and preserve order:
F1: blue(charlie)
F2: ~kind(charlie)
R1: blue(?x) & young(?x) -> kind(?x)
R2: visits(?x,rough_animal) -> sees(?x,bob)
Q: kind(charlie)
Output every numbered F/R line once and exactly one final Q. Prefix explicit negation with ~. Properties such as "Charlie is blue" become blue(charlie), never is(charlie,blue). Normalize relation predicates to their third-person schema form (chases, eats, likes, needs, sees, visits), including after "does not". Use lower_snake_case symbols. In a rule, someone/something/they/it referring to the same entity must use the same variable. Output DSL lines only."""


def resolve_generation_eos(
    model_eos_token_id: int | list[int] | None,
    tokenizer_eos_token_id: int | None,
) -> int | list[int] | None:
    """Prefer the model's generation contract, which may define multiple stops."""
    return (
        model_eos_token_id
        if model_eos_token_id is not None
        else tokenizer_eos_token_id
    )


def render_user(record: dict[str, object]) -> str:
    return f"THEORY:\n{record['theory']}\n\nQUERY:\n{record['query']}"


def theory_sentences(theory: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", theory.strip())
        if item.strip()
    ]


def is_rule_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return (
        lowered.startswith(("if ", "all "))
        or " people are " in lowered
        or " things are " in lowered
    )


def render_dsl_user(record: dict[str, object]) -> str:
    facts = rules = 0
    lines = []
    for sentence in theory_sentences(str(record["theory"])):
        if is_rule_sentence(sentence):
            rules += 1
            identifier = f"R{rules}"
        else:
            facts += 1
            identifier = f"F{facts}"
        lines.append(f"{identifier} | {sentence}")
    return "NUMBERED THEORY:\n" + "\n".join(lines) + f"\n\nQUERY | {record['query']}"


def parse_label(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    final = re.fullmatch(r"FINAL\s*:\s*(true|false|both|unknown)", lines[-1], re.I)
    return final.group(1).lower() if final else None


def extract_json(text: str) -> dict[str, object]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("compiled output is not an object")
    return value


DSL_LITERAL = re.compile(
    r"(?P<negated>~)?(?P<predicate>[a-z][a-z0-9_]*)\((?P<arguments>[^()]*)\)"
)


def parse_dsl_literal(text: str) -> dict[str, object]:
    matched = DSL_LITERAL.fullmatch(text.strip())
    if matched is None:
        raise ValueError(f"invalid DSL literal: {text!r}")
    arguments = [item.strip() for item in matched.group("arguments").split(",")]
    if not arguments or any(
        re.fullmatch(r"\??[a-z][a-z0-9_]*", item) is None for item in arguments
    ):
        raise ValueError(f"invalid DSL arguments: {text!r}")
    predicate = matched.group("predicate")
    if predicate == "is":
        raise ValueError("generic 'is' predicate is forbidden for properties")
    if predicate.startswith("is_"):
        predicate = predicate[3:]
    predicate = RELATION_CANONICAL.get(predicate, predicate)
    return {
        "predicate": predicate,
        "arguments": arguments,
        "negated": matched.group("negated") is not None,
    }


def parse_dsl(text: str) -> dict[str, object]:
    facts: list[dict[str, object]] = []
    rules: list[dict[str, object]] = []
    query = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fact_match = re.match(r"F(?P<number>\d*)\s*[:|]\s*", line)
        rule_match = re.match(r"R(?P<number>\d*)\s*[:|]\s*", line)
        query_match = re.match(r"Q\s*[:|]\s*", line)
        if fact_match is not None:
            number = fact_match.group("number") or str(len(facts) + 1)
            facts.append(
                {"id": f"f{number}", **parse_dsl_literal(line[fact_match.end() :])}
            )
        elif rule_match is not None:
            number = rule_match.group("number") or str(len(rules) + 1)
            body = line[rule_match.end() :].strip()
            if body.count("->") != 1:
                raise ValueError(f"invalid DSL rule: {line!r}")
            left, right = body.split("->")
            rules.append(
                {
                    "id": f"r{number}",
                    "premises": [parse_dsl_literal(item) for item in left.split("&")],
                    "conclusion": parse_dsl_literal(right),
                }
            )
        elif query_match is not None:
            if query is not None:
                raise ValueError("DSL must contain exactly one query")
            query = parse_dsl_literal(line[query_match.end() :])
        else:
            raise ValueError(f"unexpected DSL line: {line!r}")
    if query is None:
        raise ValueError("DSL query is missing")
    return {"facts": facts, "rules": rules, "query": query}


def compiled_literal(value: dict[str, object]) -> Literal:
    predicate = str(value["predicate"])
    arguments = tuple(str(item) for item in value["arguments"])
    negated = value["negated"]
    if not isinstance(negated, bool):
        raise ValueError("negated must be Boolean")
    if not predicate or not arguments:
        raise ValueError("literal must have a predicate and arguments")
    return Literal(predicate, arguments, negated)


def proof_payload(proof: Proof | None) -> dict[str, object] | None:
    if proof is None:
        return None
    return {
        "literal": {
            "predicate": proof.literal.predicate,
            "arguments": list(proof.literal.arguments),
            "negated": proof.literal.negated,
        },
        "source": proof.source,
        "premises": [proof_payload(item) for item in proof.premises],
    }


def execute_program(value: dict[str, object]) -> tuple[str, dict[str, object]]:
    facts = []
    for index, fact in enumerate(value["facts"]):
        facts.append((str(fact.get("id", f"f{index + 1}")), compiled_literal(fact)))
    rules = []
    for index, rule in enumerate(value["rules"]):
        rules.append(
            Rule(
                str(rule.get("id", f"r{index + 1}")),
                tuple(compiled_literal(item) for item in rule["premises"]),
                compiled_literal(rule["conclusion"]),
            )
        )
    identifiers = [identifier for identifier, _ in facts] + [rule.identifier for rule in rules]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("fact and rule ids must be globally unique")
    query = compiled_literal(value["query"])
    kb = KnowledgeBase(facts, rules).saturate()
    result = kb.query(query)
    return str(result["status"]), {
        "compiled_facts": len(facts),
        "compiled_rules": len(rules),
        "derived_literals": len(kb.proofs),
        "positive_proof": proof_payload(result["positive_proof"]),
        "negative_proof": proof_payload(result["negative_proof"]),
    }


def execute_compilation(text: str) -> tuple[str, dict[str, object]]:
    return execute_program(extract_json(text))


def execute_dsl(
    text: str,
    expected_facts: int | None = None,
    expected_rules: int | None = None,
    theory: str | None = None,
    query_text: str | None = None,
) -> tuple[str, dict[str, object]]:
    value = parse_dsl(text)
    if expected_facts is not None and len(value["facts"]) != expected_facts:
        raise ValueError(f"expected {expected_facts} facts, got {len(value['facts'])}")
    if expected_rules is not None and len(value["rules"]) != expected_rules:
        raise ValueError(f"expected {expected_rules} rules, got {len(value['rules'])}")
    if expected_facts is not None:
        observed = {str(item["id"]) for item in value["facts"]}
        expected = {f"f{index}" for index in range(1, expected_facts + 1)}
        if observed != expected:
            raise ValueError("fact identifiers do not match numbered input")
    if expected_rules is not None:
        observed = {str(item["id"]) for item in value["rules"]}
        expected = {f"r{index}" for index in range(1, expected_rules + 1)}
        if observed != expected:
            raise ValueError("rule identifiers do not match numbered input")
    if theory is not None and query_text is not None:
        validate_dsl_alignment(value, theory, query_text)
    return execute_program(value)


def expected_theory_counts(theory: str) -> tuple[int, int]:
    sentences = theory_sentences(theory)
    rules = sum(is_rule_sentence(item) for item in sentences)
    return len(sentences) - rules, rules


def _surface_contains(surface: str, symbol: str) -> bool:
    phrase = symbol.replace("_", " ")
    lowered = surface.lower()
    return phrase in lowered or (phrase.endswith("s") and phrase[:-1] in lowered)


def _validate_literal_surface(literal: dict[str, object], surface: str, identifier: str) -> None:
    if not _surface_contains(surface, str(literal["predicate"])):
        raise ValueError(f"{identifier} predicate is not aligned to its source sentence")
    for term in literal["arguments"]:
        term = str(term)
        if not term.startswith("?") and not _surface_contains(surface, term):
            raise ValueError(f"{identifier} constant {term!r} is absent from its source sentence")


def validate_dsl_alignment(
    value: dict[str, object], theory: str, query_text: str
) -> None:
    sentences = theory_sentences(theory)
    fact_sentences = [item for item in sentences if not is_rule_sentence(item)]
    rule_sentences = [item for item in sentences if is_rule_sentence(item)]
    errors = []

    def capture(function, *arguments) -> None:
        try:
            function(*arguments)
        except ValueError as error:
            errors.append(str(error))

    for item in value["facts"]:
        index = int(str(item["id"])[1:]) - 1
        capture(_validate_literal_surface, item, fact_sentences[index], str(item["id"]))
        if any(str(term).startswith("?") for term in item["arguments"]):
            errors.append(f"{item['id']} fact cannot contain variables")
    generic_pattern = re.compile(r"\b(someone|something|they|it|people|things)\b", re.I)
    for item in value["rules"]:
        index = int(str(item["id"])[1:]) - 1
        surface = rule_sentences[index]
        literals = list(item["premises"]) + [item["conclusion"]]
        for literal in literals:
            capture(_validate_literal_surface, literal, surface, str(item["id"]))
        has_variable = any(
            str(term).startswith("?")
            for literal in literals
            for term in literal["arguments"]
        )
        is_generic = generic_pattern.search(surface) is not None
        if has_variable and not is_generic:
            errors.append(f"{item['id']} must remain a ground rule for named entities")
        if is_generic and not has_variable:
            errors.append(f"{item['id']} must retain a variable for its generic entity")
    capture(_validate_literal_surface, value["query"], query_text, "query")
    if errors:
        raise ValueError("; ".join(dict.fromkeys(errors)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_repair_content(
    content: str, previous: dict[str, object], feedback: str
) -> str:
    prompt = (
        content
        + "\n\nPREVIOUS ATTEMPT (rejected):\n"
        + str(previous["raw_output"])
        + "\n\nVALIDATOR ERROR:\n"
    )
    if feedback == "specific":
        prompt += str(previous["error"])
    elif feedback == "generic":
        prompt += "The previous program failed validation."
    else:
        raise ValueError(f"unsupported repair feedback: {feedback}")
    return prompt + "\n\nReturn a complete corrected DSL program from scratch."


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("direct", "cot", "cot_visible", "compile", "compile_dsl"),
        required=True,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repair-from", type=Path)
    parser.add_argument(
        "--repair-feedback", choices=("specific", "generic"), default="specific"
    )
    args = parser.parse_args()

    if args.repair_from is not None and args.mode != "compile_dsl":
        parser.error("--repair-from is supported only for compile_dsl")
    if args.repair_from is None and args.repair_feedback != "specific":
        parser.error("--repair-feedback is meaningful only with --repair-from")

    records = [json.loads(line) for line in args.benchmark.read_text().splitlines()]
    if args.limit is not None:
        records = records[: args.limit]
    repair_rows: dict[str, dict[str, object]] = {}
    if args.repair_from is not None:
        repair_rows = {
            str(item["case_id"]): item
            for item in (
                json.loads(line) for line in args.repair_from.read_text().splitlines()
            )
        }
        records = [
            record
            for record in records
            if repair_rows.get(str(record["case_id"]), {}).get("error") is not None
        ]
    completed: dict[str, dict[str, object]] = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            item = json.loads(line)
            completed[str(item["case_id"])] = item
    pending = [record for record in records if str(record["case_id"]) not in completed]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    ).eval()
    generation_eos_token_id = resolve_generation_eos(
        getattr(model.generation_config, "eos_token_id", None),
        tokenizer.eos_token_id,
    )
    system = {
        "direct": DIRECT_SYSTEM,
        "cot": COT_SYSTEM,
        "cot_visible": COT_SYSTEM,
        "compile": COMPILE_SYSTEM,
        "compile_dsl": DSL_SYSTEM,
    }[args.mode]
    thinking = args.mode == "cot"
    default_tokens = {
        "direct": 32,
        "cot": 512,
        "cot_visible": 768,
        "compile": 1536,
        "compile_dsl": 768,
    }[args.mode]
    max_new_tokens = args.max_new_tokens or default_tokens
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.output.open("a") as handle:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            user_contents = []
            for record in batch:
                content = (
                    render_dsl_user(record)
                    if args.mode == "compile_dsl"
                    else render_user(record)
                )
                if args.repair_from is not None:
                    previous = repair_rows[str(record["case_id"])]
                    content = render_repair_content(
                        content, previous, args.repair_feedback
                    )
                user_contents.append(content)
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": (
                                content
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
                for record, content in zip(batch, user_contents, strict=True)
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            before = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=generation_eos_token_id,
                )
            latency = (time.perf_counter() - before) / len(batch)
            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for row_index, (record, input_length) in enumerate(zip(batch, lengths, strict=True)):
                output_tokens = generated[row_index, inputs["input_ids"].shape[1] :]
                text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                error = None
                metadata: dict[str, object] = {}
                if args.mode in {"compile", "compile_dsl"}:
                    try:
                        if args.mode == "compile_dsl":
                            expected_facts, expected_rules = expected_theory_counts(
                                str(record["theory"])
                            )
                            prediction, metadata = execute_dsl(
                                text,
                                expected_facts,
                                expected_rules,
                                str(record["theory"]),
                                str(record["query"]),
                            )
                        else:
                            prediction, metadata = execute_compilation(text)
                    except Exception as exception:
                        prediction = None
                        error = f"{type(exception).__name__}: {exception}"
                else:
                    prediction = parse_label(text)
                    if prediction is None:
                        error = "label_parse_failure"
                result = {
                    "case_id": record["case_id"],
                    "mode": args.mode,
                    "prediction": prediction,
                    "gold": record["label"],
                    "correct": prediction == record["label"],
                    "error": error,
                    "input_tokens": int(input_length),
                    "output_tokens": int((output_tokens != tokenizer.pad_token_id).sum()),
                    "latency_seconds": latency,
                    "raw_output": text,
                    "attempt": 2 if args.repair_from is not None else 1,
                    "previous_error": (
                        repair_rows[str(record["case_id"])].get("error")
                        if args.repair_from is not None
                        else None
                    ),
                    **metadata,
                }
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                handle.flush()
            print(args.mode, offset + len(batch), "/", len(pending), flush=True)

    protocol = {
        "model": str(args.model),
        "benchmark": str(args.benchmark),
        "benchmark_sha256": file_sha256(args.benchmark),
        "mode": args.mode,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_new_tokens": max_new_tokens,
        "thinking": thinking,
        "generation_eos_token_id": generation_eos_token_id,
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "records_requested": len(records),
        "repair_from": str(args.repair_from) if args.repair_from is not None else None,
        "repair_feedback": args.repair_feedback if args.repair_from is not None else None,
    }
    args.protocol_output.parent.mkdir(parents=True, exist_ok=True)
    args.protocol_output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
