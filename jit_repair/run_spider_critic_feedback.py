"""Run Llama with frozen execution and gold-free consistency-critic feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from .run_spider_feedback import load_jsonl, resolve_generation_eos
from .spider_critic_feedback import critic_score, lint_question_sql, render_critic_specific_feedback
from .spider_feedback import (
    SQL_SYSTEM_PROMPT,
    execute_readonly,
    extract_sql,
    file_sha256,
    locate_database,
    render_generic_feedback,
    render_initial_user,
    render_repair_user,
    schema_snapshot,
)


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path, required=True)
    parser.add_argument("--repair-from", type=Path)
    parser.add_argument("--feedback", choices=("specific", "generic"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if (args.repair_from is None) != (args.feedback is None):
        parser.error("--repair-from and --feedback must be provided together")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    benchmark_rows = load_jsonl(args.benchmark)
    if args.limit is not None:
        benchmark_rows = benchmark_rows[: args.limit]
    expected_fields = {"case_id", "source_index", "db_id", "question"}
    for row in benchmark_rows:
        if set(row) != expected_fields:
            raise ValueError(f"gold-blind benchmark fields differ from frozen contract: {sorted(row)}")

    previous: dict[str, dict[str, Any]] = {}
    requested = benchmark_rows
    if args.repair_from is not None:
        previous = {str(row["case_id"]): row for row in load_jsonl(args.repair_from)}
        if set(previous) != {str(row["case_id"]) for row in benchmark_rows}:
            raise ValueError("repair source and benchmark case identifiers differ")
        requested = [row for row in benchmark_rows if bool(previous[str(row["case_id"])]["needs_repair"])]
    completed = (
        {str(row["case_id"]): row for row in load_jsonl(args.output)} if args.output.exists() else {}
    )
    pending = [row for row in requested if str(row["case_id"]) not in completed]

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
    eos_token_id = resolve_generation_eos(
        getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id
    )

    snapshots: dict[str, tuple[str, tuple[str, ...]]] = {}
    for row in pending:
        db_id = str(row["db_id"])
        if db_id not in snapshots:
            snapshots[db_id] = schema_snapshot(locate_database(args.database_dir, db_id))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    max_input_tokens = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            contents: list[str] = []
            for row in batch:
                schema, identifiers = snapshots[str(row["db_id"])]
                if args.repair_from is None:
                    content = render_initial_user(str(row["question"]), schema)
                else:
                    prior = previous[str(row["case_id"])]
                    feedback = (
                        render_critic_specific_feedback(
                            prior.get("error"), list(prior["critic_issues"]), identifiers
                        )
                        if args.feedback == "specific"
                        else render_generic_feedback()
                    )
                    content = render_repair_user(
                        str(row["question"]), schema, str(prior["sql"]), feedback
                    )
                contents.append(content)
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SQL_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for content in contents
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            max_input_tokens = max(max_input_tokens, *(int(value) for value in lengths))
            before = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                )
            latency = (time.perf_counter() - before) / len(batch)
            for index, (row, input_tokens) in enumerate(zip(batch, lengths, strict=True)):
                output_tokens = generated[index, inputs["input_ids"].shape[1] :]
                raw = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                sql = extract_sql(raw)
                execution = execute_readonly(
                    locate_database(args.database_dir, str(row["db_id"])),
                    sql,
                    args.timeout_seconds,
                )
                issues = lint_question_sql(str(row["question"]), sql) if execution.valid else []
                prior = previous.get(str(row["case_id"]))
                result = {
                    "case_id": row["case_id"],
                    "source_index": row["source_index"],
                    "db_id": row["db_id"],
                    "sql": sql,
                    "raw_output": raw,
                    "execution_valid": execution.valid,
                    "error": execution.error,
                    "error_class": execution.error_class,
                    "critic_issues": issues,
                    "critic_score": critic_score(issues),
                    "needs_repair": (not execution.valid) or bool(issues),
                    "rows_observed_capped": execution.rows_observed,
                    "execution_seconds": execution.elapsed_seconds,
                    "input_tokens": int(input_tokens),
                    "output_tokens": int((output_tokens != tokenizer.pad_token_id).sum()),
                    "generation_latency_seconds": latency,
                    "attempt": 1 if prior is None else int(prior["attempt"]) + 1,
                    "previous_error_class": prior.get("error_class") if prior else None,
                    "previous_critic_score": prior.get("critic_score") if prior else None,
                    "feedback": args.feedback,
                    "gold_visible_to_generation": False,
                }
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            print(f"spider-critic {args.feedback or 'initial'} {offset + len(batch)}/{len(pending)}", flush=True)

    protocol = {
        "protocol_id": "spider-gold-free-critic-feedback-runner-v1",
        "model": str(args.model),
        "benchmark": str(args.benchmark),
        "benchmark_sha256": file_sha256(args.benchmark),
        "database_dir": str(args.database_dir),
        "records_in_benchmark": len(benchmark_rows),
        "records_requested": len(requested),
        "records_generated_this_invocation": len(pending),
        "repair_from": str(args.repair_from) if args.repair_from else None,
        "repair_from_sha256": file_sha256(args.repair_from) if args.repair_from else None,
        "feedback": args.feedback,
        "retry_policy": "execution_failure_or_gold_free_consistency_issue",
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "maximum_observed_input_tokens": max_input_tokens,
        "seed": args.seed,
        "do_sample": False,
        "timeout_seconds": args.timeout_seconds,
        "system_prompt_sha256": hashlib.sha256(SQL_SYSTEM_PROMPT.encode()).hexdigest(),
        "gold_visible_to_generation": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.protocol_output.parent.mkdir(parents=True, exist_ok=True)
    args.protocol_output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
