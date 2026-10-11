"""GPU driver: one vLLM engine per model; runs the C2 SQL arms
(greedy, JIT-Repair = greedy + frozen generic repair + frozen selector, 8 samples) and the C1/T1
ProofWriter arms (direct, CoT, Qwen3 thinking, DSL + specific/generic repair x3). Every unit is
written as it completes; resumable. Gold is never used here except to fill the frozen-format
'correct' field of ProofWriter records (as the frozen runner did); gold never affects generation,
checking, repair triggering or selection."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402

import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INBOX = jp.INBOX
OUT = jp.OUTBOX
DB = {"spider": (jp.DATA / "spider/spider_data/test_database"),
      "bird": (jp.DATA / "bird/dev_20240627/dev_databases")}
SAFE = {"spider": INBOX / "frozen/spider_llama31/prepared/safe.jsonl",
        "bird": INBOX / "frozen/bird_assets/safe.jsonl"}
MAXTOK = {"spider": 512, "bird": 768}
PW = {"conf": INBOX / "data/proofwriter/proofwriter_fourval_test_1000.jsonl",
      "d5": INBOX / "data/proofwriter/logiclm_d5_600.jsonl"}
MODELS = {"llama31_8b": str(jp.MODELS / "llama31_8b"), "mistral7b": str(jp.MODELS / "mistral7b"),
          "qwen3_8b": str(jp.MODELS / "qwen3_8b"), "qwen3_14b": str(jp.MODELS / "qwen3_14b")}
PW_SETS = {"qwen3_14b": ["conf", "d5"], "qwen3_8b": ["d5"], "llama31_8b": ["d5"], "mistral7b": ["d5"]}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wjl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, p)


def _check(task):
    import sql_tools
    bench, row, raw = task
    return sql_tools.check(bench, row, raw, DB[bench])


def sample_tokens(s):
    return s.get("output_tokens_frozen_convention", s["output_tokens"])


# ------------------------------------------------------------------ SQL (C2 + JIT regen)
def run_sql(eng, tag, pool, benches):
    import sql_tools
    from jit_repair import spider_feedback as sf, bird_minidev_feedback as bf
    from jit_repair.merge_spider_critic_retries import choose
    for bench in benches:
        d = OUT / "c2" / bench / tag
        safe = jl(SAFE[bench]); by = {str(r["case_id"]): r for r in safe}
        prompts = sql_tools.build_prompts(bench, safe, DB[bench])
        pmap = {p["id"]: p for p in prompts}
        mt = MAXTOK[bench]
        if not (d / "final.jsonl").exists():
            g = eng.run(prompts, str(d / "gen_initial.jsonl"), temperature=0.0, max_tokens=mt, tag=f"{tag}/{bench}/initial")
            chk = pool.map(_check, [(bench, by[p["id"]], g[p["id"]]["samples"][0]["text"]) for p in prompts], chunksize=8)
            initial = []
            for p, c in zip(prompts, chk):
                s = g[p["id"]]["samples"][0]
                initial.append({"case_id": p["id"], "db_id": p["db_id"], "raw_output": s["text"], **c,
                                "input_tokens": g[p["id"]]["input_tokens"], "output_tokens": sample_tokens(s),
                                "finish_reason": s["finish_reason"], "attempt": 1, "feedback": None,
                                "previous_error_class": None, "previous_critic_score": None,
                                "gold_visible_to_generation": False, "stack": "vllm-a100"})
            wjl(d / "initial.jsonl", initial)
            flagged = [r for r in initial if r["needs_repair"]]
            rrows = []
            for r in flagged:
                row = by[r["case_id"]]
                if bench == "spider":
                    schema = sf.schema_snapshot(sf.locate_database(DB[bench], str(row["db_id"])))[0]
                    user = sf.render_repair_user(str(row["question"]), schema, str(r["sql"]), sf.render_generic_feedback())
                else:
                    user = bf.render_repair_user(str(row["question"]), str(row["evidence"]), str(row["schema"]),
                                                 str(r["sql"]), bf.render_generic_feedback())
                rrows.append({"id": r["case_id"], "system": pmap[r["case_id"]]["system"], "user": user})
            gr = eng.run(rrows, str(d / "gen_repair_1.jsonl"), temperature=0.0, max_tokens=mt, tag=f"{tag}/{bench}/repair") if rrows else {}
            chk = pool.map(_check, [(bench, by[x["id"]], gr[x["id"]]["samples"][0]["text"]) for x in rrows], chunksize=8)
            repairs = {}
            for x, c, prev in zip(rrows, chk, flagged):
                s = gr[x["id"]]["samples"][0]
                repairs[x["id"]] = {"case_id": x["id"], "db_id": prev["db_id"], "raw_output": s["text"], **c,
                                    "input_tokens": gr[x["id"]]["input_tokens"], "output_tokens": sample_tokens(s),
                                    "finish_reason": s["finish_reason"], "attempt": 2, "feedback": "generic",
                                    "previous_error_class": prev["error_class"], "previous_critic_score": prev["critic_score"],
                                    "gold_visible_to_generation": False, "stack": "vllm-a100"}
            wjl(d / "repair_1.jsonl", [repairs[x["id"]] for x in rrows])
            final = []
            for r in initial:
                if r["case_id"] in repairs:
                    sel, reason = choose(r, repairs[r["case_id"]])
                else:
                    sel, reason = r, "unflagged_previous_immutable"
                sel = dict(sel); sel["selection_reason"] = reason
                sel["jit_calls"] = 1 + int(r["case_id"] in repairs)
                sel["jit_output_tokens"] = r["output_tokens"] + (repairs[r["case_id"]]["output_tokens"] if r["case_id"] in repairs else 0)
                sel["jit_input_tokens"] = r["input_tokens"] + (repairs[r["case_id"]]["input_tokens"] if r["case_id"] in repairs else 0)
                final.append(sel)
            wjl(d / "final.jsonl", final)
        eng.run(prompts, str(d / "gen_samples.jsonl"), n=8, temperature=0.7, top_p=0.95, max_tokens=mt,
                seed=20261011, tag=f"{tag}/{bench}/samples")
        (d / "GEN_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


# ------------------------------------------------------------------ ProofWriter (C1 + T1)
def pw_records(bench_rows, gen, mode, parse):
    import pw_tools as pt
    out = []
    for rec in bench_rows:
        g = gen[rec["case_id"]]; s = g["samples"][0]
        if mode == "dsl":
            pred, err, meta = pt.validate_dsl(rec, s["text"])
        else:
            pred = parse(s["text"]); err = None if pred is not None else "label_parse_failure"; meta = {}
        r = {"case_id": rec["case_id"], "mode": mode, "prediction": pred, "gold": rec["label"],
             "correct": pred == rec["label"], "error": err, "input_tokens": g["input_tokens"],
             "output_tokens": sample_tokens(s), "raw_output": s["text"], "finish_reason": s["finish_reason"],
             "attempt": 1, "previous_error": None, "stack": "vllm-a100", **meta}
        if mode in ("cot", "think"):
            r["strict_prediction"] = pt.strict_label(s["text"])
        out.append(r)
    return out


def run_reasoning_arm(eng, tag, d, rows, bench, mode, base_tokens, ext_tokens, sampling, thinking):
    import pw_tools as pt
    if (d / f"{mode}.jsonl").exists():
        return
    parse = pt.extract_label
    g = eng.run(rows, str(d / f"gen_{mode}.jsonl"), max_tokens=base_tokens, enable_thinking=thinking,
                tag=f"{tag}/{mode}", **sampling)
    trunc = [r for r in rows if g[r["id"]]["samples"][0]["finish_reason"] == "length"]
    info = {"n": len(rows), "truncated_at_base": len(trunc), "base_max_tokens": base_tokens}
    if len(trunc) / len(rows) >= 0.02:
        # ONE declared increase, truncated items only, same seeds (pre-registered)
        g2 = eng.run(trunc, str(d / f"gen_{mode}_ext.jsonl"), max_tokens=ext_tokens, enable_thinking=thinking,
                     tag=f"{tag}/{mode}-ext", **sampling)
        for r in trunc:
            g[r["id"]] = g2[r["id"]]
        info["declared_increase_to"] = ext_tokens
        info["truncated_after_increase"] = sum(g[r["id"]]["samples"][0]["finish_reason"] == "length" for r in rows)
    wjl(d / f"{mode}.jsonl", pw_records(bench, g, mode, parse))
    json.dump(info, open(d / f"{mode}.truncation.json", "w"), indent=1)


def run_pw(eng, tag, sets):
    import pw_tools as pt
    for st in sets:
        d = OUT / "pw" / st / tag
        bench = jl(PW[st]); byid = {r["case_id"]: r for r in bench}
        # DSL initial + specific/generic repair x3 (JIT arm for C1; T1 arms)
        if not (d / "dsl_initial.jsonl").exists():
            g = eng.run([pt.dsl_row(r) for r in bench], str(d / "gen_dsl_initial.jsonl"), max_tokens=768, tag=f"{tag}/{st}/dsl")
            wjl(d / "dsl_initial.jsonl", pw_records(bench, g, "dsl", None))
        initial = {r["case_id"]: r for r in jl(d / "dsl_initial.jsonl")}
        for fb in ("specific", "generic"):
            prev = initial
            for k in (1, 2, 3):
                mpath = d / f"dsl_{fb}_merged_{k}.jsonl"
                if mpath.exists():
                    prev = {r["case_id"]: r for r in jl(mpath)}; continue
                todo = [byid[c] for c, r in prev.items() if r["error"] is not None]
                rows = [pt.repair_row(rec, prev[rec["case_id"]], fb) for rec in todo]
                g = eng.run(rows, str(d / f"gen_dsl_{fb}_{k}.jsonl"), max_tokens=768, tag=f"{tag}/{st}/{fb}{k}") if rows else {}
                reps = {}
                for rec in todo:
                    r = pw_records([rec], g, "dsl", None)[0]
                    r["attempt"] = 2; r["previous_error"] = prev[rec["case_id"]]["error"]; r["feedback"] = fb
                    reps[rec["case_id"]] = r
                wjl(d / f"dsl_{fb}_repair_{k}.jsonl", list(reps.values()))
                prev = pt.merge(prev, reps)
                wjl(mpath, [prev[r["case_id"]] for r in bench])
        # reasoning baselines
        run_reasoning_arm(eng, tag, d, [pt.cot_row(r) for r in bench], bench, "cot", 2048, 4096,
                          {"temperature": 0.0}, "false")
        if tag.startswith("qwen3"):
            run_reasoning_arm(eng, tag, d, [pt.direct_row(r) for r in bench], bench, "think", 8192, 16384,
                              {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 20261011}, "true")
        if not (d / "direct.jsonl").exists():
            g = eng.run([pt.direct_row(r) for r in bench], str(d / "gen_direct.jsonl"), max_tokens=32, tag=f"{tag}/{st}/direct")
            wjl(d / "direct.jsonl", pw_records(bench, g, "direct", pt.strict_label))
        (d / "GEN_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--tasks", default="sql,pw")
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()
    from gen_vllm import Engine
    pool = mp.get_context("spawn").Pool(48)
    eng = Engine(MODELS[a.model_tag], a.model_tag, max_model_len=a.max_model_len)
    t0 = time.time()
    if "sql" in a.tasks:
        run_sql(eng, a.model_tag, pool, ["spider", "bird"])
    if "pw" in a.tasks:
        run_pw(eng, a.model_tag, PW_SETS[a.model_tag])
    print(f"DRIVER_DONE {a.model_tag} {time.time() - t0:.0f}s", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
