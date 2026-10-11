"""Teacher-forced confidence of each GREEDY output under its own model (gold-free): sum / mean / min token
log-prob of the output string (re-tokenised, no special tokens) given the frozen prompt. Used for the
confidence-gate comparator in the allocation study (X7)."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = jp.OUTBOX
MODELS = {"llama31_8b": str(jp.MODELS / "llama31_8b"), "mistral7b": str(jp.MODELS / "mistral7b"),
          "qwen3_8b": str(jp.MODELS / "qwen3_8b"), "qwen3_14b": str(jp.MODELS / "qwen3_14b")}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cell_rows(cell, model):
    """Return [(id, system, user, greedy_text)] for a cell."""
    if cell in ("spider", "bird"):
        import sql_tools
        from generate_c1_c2 import SAFE, DB
        safe = jl(SAFE[cell]); prompts = {p["id"]: p for p in sql_tools.build_prompts(cell, safe, DB[cell])}
        ini = jl(OUT / "c2" / cell / model / "initial.jsonl")
    elif cell == "drspider":
        from jit_repair import drspider_oneuse as do
        safe = jl(str(jp.INBOX / "drspider/safe_subset.jsonl"))
        prompts = {r["case_id"]: {"system": do.DRSPIDER_SQL_SYSTEM_PROMPT, "user": do.render_initial_user(str(r["question"]), str(r["schema"]))} for r in safe}
        ini = jl(OUT / "x3b" / model / "initial.jsonl")
    elif cell == "birddev":
        from jit_repair import bird_minidev_feedback as bf
        safe = jl(str(jp.OUTBOX / "x7/birddev_assets/safe.jsonl"))
        prompts = {r["case_id"]: {"system": bf.BIRD_SQL_SYSTEM_PROMPT, "user": bf.render_initial_user(str(r["question"]), str(r["evidence"]), str(r["schema"]))} for r in safe}
        ini = jl(OUT / "x7" / "birddev" / model / "initial.jsonl")
    else:
        raise ValueError(cell)
    return [(r["case_id"], prompts[r["case_id"]]["system"], prompts[r["case_id"]]["user"], r["raw_output"]) for r in ini]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--cells", required=True)
    ap.add_argument("--max-model-len", type=int, default=12288)
    # engineering (not method): prompt_logprobs materialises full-vocabulary logits per prefill chunk; with Qwen3's
    # 152k vocabulary and long BIRD prompts this OOMs at 0.9 utilisation (17:12Z). Values are unaffected by batching.
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(MODELS[a.model], local_files_only=True)
    llm = LLM(model=MODELS[a.model], dtype="bfloat16", seed=20261011, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              max_num_batched_tokens=a.max_num_batched_tokens)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    for cell in a.cells.split(","):
        out = OUT / "x7" / "logprobs" / f"{cell}__{a.model}.jsonl"
        if out.exists():
            continue
        rows = cell_rows(cell, a.model)
        reqs, meta = [], []
        for cid, system, user, text in rows:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            ptxt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            pids = tok(ptxt)["input_ids"]; oids = tok(text, add_special_tokens=False)["input_ids"]
            if not oids:
                oids = []
            reqs.append({"prompt_token_ids": pids + oids}); meta.append((cid, len(pids), oids))
        res = llm.generate(reqs, sp, use_tqdm=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as h:
            for (cid, np_, oids), r in zip(meta, res):
                lps = []
                for i, t in enumerate(oids):
                    d = r.prompt_logprobs[np_ + i]
                    lps.append(d[t].logprob if d is not None and t in d else float("nan"))
                n = len(lps)
                h.write(json.dumps({"case_id": cid, "n_tok": n, "sum_lp": sum(lps) if n else 0.0,
                                    "mean_lp": (sum(lps) / n) if n else 0.0, "min_lp": min(lps) if n else 0.0}) + "\n")
        print("LP", cell, a.model, len(rows), flush=True)


if __name__ == "__main__":
    main()
