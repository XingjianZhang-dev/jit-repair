"""Run one gold-blind generation stage for the Dr.Spider OOD one-use experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from .drspider_oneuse import (
    DRSPIDER_SQL_SYSTEM_PROMPT,
    SAFE_FIELDS,
    SELF_REFINE_FEEDBACK_SYSTEM_PROMPT,
    execute_readonly,
    extract_sql,
    file_sha256,
    locate_drspider_database,
    render_generic_repair_user,
    render_initial_user,
    render_self_refine_feedback_user,
    render_self_refine_iterate_user,
    schema_snapshot,
)
from .run_spider_feedback import load_jsonl, resolve_generation_eos
from .spider_critic_feedback import critic_score, lint_question_sql


MODES = {"initial", "generic_repair", "self_refine_feedback", "self_refine_iterate"}


def _id_hash(rows: list[dict[str, Any]]) -> str:
    payload = "".join(f"{row['case_id']}\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_keyed(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    keyed = {str(row["case_id"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate case identifier in {path}")
    return keyed


def _requested_rows(
    mode: str,
    benchmark_rows: list[dict[str, Any]],
    initial: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if mode == "initial":
        return benchmark_rows
    if set(initial) != {str(row["case_id"]) for row in benchmark_rows}:
        raise ValueError("initial predictions do not cover the full Dr.Spider benchmark")
    if mode == "generic_repair":
        return [
            row
            for row in benchmark_rows
            if bool(row["in_ablation_subset"]) or bool(initial[str(row["case_id"])]["needs_repair"])
        ]
    return [row for row in benchmark_rows if bool(row["in_ablation_subset"])]


def _render_stage_content(
    mode: str,
    row: dict[str, Any],
    initial: dict[str, dict[str, Any]],
    feedback: dict[str, dict[str, Any]],
) -> str:
    case_id = str(row["case_id"])
    if mode == "initial":
        return render_initial_user(str(row["question"]), str(row["schema"]))
    if mode == "generic_repair":
        return render_generic_repair_user(
            str(row["question"]), str(row["schema"]), str(initial[case_id]["sql"])
        )
    if mode == "self_refine_feedback":
        return render_self_refine_feedback_user(
            str(row["question"]), str(row["schema"]), str(initial[case_id]["sql"])
        )
    return render_self_refine_iterate_user(
        str(row["question"]),
        str(row["schema"]),
        str(initial[case_id]["sql"]),
        str(feedback[case_id]["feedback_text"]),
    )


def main() -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--feedback-from", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--feedback-max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--context-limit", type=int)
    args = parser.parse_args()

    if args.mode != "initial" and args.initial is None:
        raise ValueError(f"{args.mode} requires --initial")
    if args.mode == "initial" and (args.initial is not None or args.feedback_from is not None):
        raise ValueError("initial mode cannot consume earlier model outputs")
    if args.mode == "self_refine_iterate" and args.feedback_from is None:
        raise ValueError("self_refine_iterate requires --feedback-from")
    if args.mode != "self_refine_iterate" and args.feedback_from is not None:
        raise ValueError("only self_refine_iterate may consume --feedback-from")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    benchmark_rows = load_jsonl(args.benchmark)
    if len(benchmark_rows) != 15_269:
        raise ValueError("formal Dr.Spider run requires exactly 15,269 rows")
    if any(set(row) != SAFE_FIELDS for row in benchmark_rows):
        raise ValueError("gold-blind Dr.Spider benchmark fields differ from frozen contract")
    initial = _load_keyed(args.initial)
    feedback = _load_keyed(args.feedback_from)
    requested = _requested_rows(args.mode, benchmark_rows, initial)
    if args.mode == "self_refine_iterate" and set(feedback) != {
        str(row["case_id"]) for row in requested
    }:
        raise ValueError("Self-Refine feedback does not exactly cover the frozen subset")
    completed = _load_keyed(args.output) if args.output.exists() else {}
    requested_ids = {str(row["case_id"]) for row in requested}
    if not set(completed).issubset(requested_ids):
        raise ValueError("resumed output contains cases outside the requested stage")
    pending = [row for row in requested if str(row["case_id"]) not in completed]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    max_new_tokens = (
        args.feedback_max_new_tokens if args.mode == "self_refine_feedback" else args.max_new_tokens
    )
    system_prompt = (
        SELF_REFINE_FEEDBACK_SYSTEM_PROMPT
        if args.mode == "self_refine_feedback"
        else DRSPIDER_SQL_SYSTEM_PROMPT
    )

    def prompt_for(row: dict[str, Any]) -> str:
        content = _render_stage_content(args.mode, row, initial, feedback)
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    configured_context = int(getattr(config, "max_position_embeddings", 0) or 0)
    if args.context_limit is not None:
        context_limit = args.context_limit
        if configured_context and context_limit > configured_context:
            raise ValueError("explicit context limit exceeds model max_position_embeddings")
    else:
        context_limit = configured_context
    if context_limit <= 0:
        raise ValueError("unable to resolve a finite model context limit")
    # Every prompt in the requested stage is checked before any output from that
    # stage is generated. This also makes resumed stages fail closed.
    preflight_max_input_tokens = 0
    for offset in range(0, len(requested), 128):
        prompts = [prompt_for(row) for row in requested[offset : offset + 128]]
        token_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        preflight_max_input_tokens = max(
            preflight_max_input_tokens, *(len(values) for values in token_ids)
        )
    if preflight_max_input_tokens + max_new_tokens > context_limit:
        raise ValueError(
            "stage-wide tokenizer preflight exceeds the frozen context limit: "
            f"{preflight_max_input_tokens}+{max_new_tokens}>{context_limit}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    ).eval()
    eos_token_id = resolve_generation_eos(
        getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    max_input_tokens = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            prompts = [prompt_for(row) for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            max_input_tokens = max(max_input_tokens, *(int(value) for value in lengths))
            before = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                )
            batch_elapsed = time.perf_counter() - before
            latency = batch_elapsed / len(batch)
            for index, (row, input_tokens) in enumerate(zip(batch, lengths, strict=True)):
                case_id = str(row["case_id"])
                output_tokens = generated[index, inputs["input_ids"].shape[1] :]
                raw = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                common = {
                    "case_id": case_id,
                    "source_index": row["source_index"],
                    "set_index": row["set_index"],
                    "set_name": row["set_name"],
                    "axis": row["axis"],
                    "q_id_spider_dev": row["q_id_spider_dev"],
                    "db_id": row["db_id"],
                    "input_tokens": int(input_tokens),
                    "output_tokens": int((output_tokens != tokenizer.pad_token_id).sum()),
                    "generation_latency_seconds": latency,
                    "gold_visible_to_generation": False,
                }
                if args.mode == "self_refine_feedback":
                    result = {
                        **common,
                        "raw_output": raw,
                        "feedback_text": raw or "No specific error identified; recheck the proposed SQL.",
                        "self_refine_official_pattern": "generate_feedback_then_iterate",
                    }
                else:
                    sql = extract_sql(raw)
                    database = locate_drspider_database(
                        args.data_root, str(row["set_name"]), str(row["db_id"])
                    )
                    execution = execute_readonly(database, sql, args.timeout_seconds)
                    issues = (
                        lint_question_sql(str(row["question"]), sql) if execution.valid else []
                    )
                    result = {
                        **common,
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
                        "attempt": 1 if args.mode == "initial" else 2,
                        "previous_error_class": initial.get(case_id, {}).get("error_class"),
                        "previous_critic_score": initial.get(case_id, {}).get("critic_score"),
                        "feedback": (
                            None
                            if args.mode == "initial"
                            else "generic_checklist"
                            if args.mode == "generic_repair"
                            else "self_refine_model_feedback"
                        ),
                    }
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            processed = offset + len(batch)
            if processed == len(pending) or processed % 100 < args.batch_size:
                print(f"drspider {args.mode} {processed}/{len(pending)}", flush=True)

    protocol = {
        "protocol_id": "drspider-gold-free-oneuse-runner-v1",
        "mode": args.mode,
        "model": str(args.model),
        "benchmark": str(args.benchmark),
        "benchmark_sha256": file_sha256(args.benchmark),
        "data_root": str(args.data_root),
        "records_in_benchmark": len(benchmark_rows),
        "records_requested": len(requested),
        "requested_case_ids_sha256": _id_hash(requested),
        "records_generated_this_invocation": len(pending),
        "initial": str(args.initial) if args.initial else None,
        "initial_sha256": file_sha256(args.initial) if args.initial else None,
        "feedback_from": str(args.feedback_from) if args.feedback_from else None,
        "feedback_from_sha256": file_sha256(args.feedback_from) if args.feedback_from else None,
        "batch_size": args.batch_size,
        "max_new_tokens": max_new_tokens,
        "maximum_observed_input_tokens": max_input_tokens,
        "stage_preflight_maximum_input_tokens": preflight_max_input_tokens,
        "model_configured_context_limit": configured_context,
        "effective_context_limit": context_limit,
        "stage_preflight_plus_generation_within_context": (
            preflight_max_input_tokens + max_new_tokens <= context_limit
        ),
        "seed": args.seed,
        "do_sample": False,
        "timeout_seconds": args.timeout_seconds,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "self_refine_official_repository_commit": "9a206d41e5d2d0c241bb441f41eeadb945afaa55",
        "gold_visible_to_generation": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.protocol_output.parent.mkdir(parents=True, exist_ok=True)
    args.protocol_output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
