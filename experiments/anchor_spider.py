"""G1 output-identity check (Supplement S6): regenerate direct greedy Spider-test outputs for the first 200
questions with Llama-3.1-8B using the frozen prompt/template/decoding/max-new-tokens.
Gate per stack: >=95% of raw outputs string-identical to the frozen outputs AND slice
accuracy within +/-1 point of the frozen slice accuracy. Gate is never loosened."""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sql_tools  # noqa: E402

def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def _score(task):
    return sql_tools.score_spider(*task)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)          # frozen spider_llama31 dir
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--test-suite", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stacks", default="hf,vllm")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fz = Path(a.frozen)
    safe = jl(fz / "prepared/safe.jsonl")
    gold = [l.rstrip("\n").rsplit("\t", 1) for l in open(fz / "prepared/gold.sql", encoding="utf-8")]
    frozen_init = {r["case_id"]: r for r in jl(fz / "initial/initial.jsonl")}
    frozen_eval = {r["case_id"]: r for r in jl(fz / "evaluation/initial.cases.jsonl")}
    sl = safe[: a.n]
    ids = [r["case_id"] for r in sl]

    # prompt identity: rendered token counts vs frozen input_tokens for ALL items
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    prompts = sql_tools.build_prompts("spider", safe, Path(a.db_dir))
    mism = []
    for pr in prompts:
        text = tok.apply_chat_template([{"role": "system", "content": pr["system"]}, {"role": "user", "content": pr["user"]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        n = len(tok(text)["input_ids"])
        if n != frozen_init[pr["id"]]["input_tokens"]:
            mism.append((pr["id"], n, frozen_init[pr["id"]]["input_tokens"]))
    prompt_check = {"items": len(prompts), "input_token_mismatches": len(mism), "examples": mism[:5]}
    json.dump(prompt_check, open(out / "prompt_identity.json", "w"), indent=1)
    print("PROMPT_IDENTITY", prompt_check, flush=True)

    results = {}
    for stack in a.stacks.split(","):
        sd = out / stack; sd.mkdir(exist_ok=True)
        t0 = time.time()
        if stack == "hf":
            if not (sd / "initial.jsonl").exists() or len(jl(sd / "initial.jsonl")) < a.n:
                cmd = [sys.executable, "-m", "jit_repair.run_spider_critic_feedback", "--model", a.model,
                       "--benchmark", str(fz / "prepared/safe.jsonl"), "--limit", str(a.n), "--database-dir", a.db_dir,
                       "--output", str(sd / "initial.jsonl"), "--protocol-output", str(sd / "protocol.json"),
                       "--batch-size", "8", "--max-new-tokens", "512"]
                subprocess.run(cmd, check=True)
            recs = {r["case_id"]: r for r in jl(sd / "initial.jsonl")}
        else:
            from gen_vllm import Engine
            eng = Engine(a.model, "llama31_8b", max_model_len=4096)
            gen = eng.run([p for p in prompts if p["id"] in set(ids)], str(sd / "generations.jsonl"),
                          temperature=0.0, max_tokens=512, tag="anchor-vllm")
            recs = {}
            for r in sl:
                raw = gen[r["case_id"]]["samples"][0]["text"]
                rec = sql_tools.check("spider", r, raw, Path(a.db_dir)); rec["raw_output"] = raw
                rec["output_tokens"] = gen[r["case_id"]]["samples"][0]["output_tokens"]
                recs[r["case_id"]] = rec
            del eng
            import gc; gc.collect()
        gen_s = time.time() - t0
        tasks = [(a.test_suite, a.db_dir, r["db_id"], recs[r["case_id"]]["sql"], gold[i][0], bool(recs[r["case_id"]]["execution_valid"]))
                 for i, r in enumerate(sl)]
        with ProcessPoolExecutor(32) as ex:
            scores = list(ex.map(_score, tasks))
        same_raw = sum(recs[i]["raw_output"] == frozen_init[i]["raw_output"] for i in ids)
        same_sql = sum(recs[i]["sql"] == frozen_init[i]["sql"] for i in ids)
        acc_new = sum(s for s, _ in scores) / a.n
        acc_frozen = sum(bool(frozen_eval[i]["correct"]) for i in ids) / a.n
        # sanity: re-score the FROZEN sql with the same evaluator (instrument check)
        ftasks = [(a.test_suite, a.db_dir, r["db_id"], frozen_init[r["case_id"]]["sql"], gold[i][0], bool(frozen_init[r["case_id"]]["execution_valid"]))
                  for i, r in enumerate(sl)]
        with ProcessPoolExecutor(32) as ex:
            fscores = list(ex.map(_score, ftasks))
        acc_frozen_rescored = sum(s for s, _ in fscores) / a.n
        rescore_agree = sum(int(s == bool(frozen_eval[i]["correct"])) for (s, _), i in zip(fscores, ids))
        diffs = [{"case_id": i, "frozen": frozen_init[i]["raw_output"], "new": recs[i]["raw_output"]} for i in ids
                 if recs[i]["raw_output"] != frozen_init[i]["raw_output"]]
        res = {"stack": stack, "n": a.n, "identical_raw": same_raw, "identical_raw_rate": same_raw / a.n,
               "identical_sql": same_sql, "identical_sql_rate": same_sql / a.n,
               "accuracy_new": acc_new, "accuracy_frozen": acc_frozen,
               "accuracy_frozen_rescored_here": acc_frozen_rescored, "frozen_rescore_case_agreement": rescore_agree,
               "accuracy_abs_diff_points": abs(acc_new - acc_frozen) * 100,
               "evaluator_errors_new": sum(e is not None for _, e in scores),
               "gate_identity_ge_95pct": same_raw / a.n >= 0.95,
               "gate_accuracy_within_1pt": abs(acc_new - acc_frozen) * 100 <= 1.0 + 1e-9,
               "seconds": gen_s}
        res["GATE_PASS"] = bool(res["gate_identity_ge_95pct"] and res["gate_accuracy_within_1pt"])
        json.dump({"result": res, "diffs": diffs}, open(sd / "gate.json", "w"), indent=1, ensure_ascii=False)
        results[stack] = res
        print("ANCHOR", json.dumps(res), flush=True)
    json.dump(results, open(out / "anchor_summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
