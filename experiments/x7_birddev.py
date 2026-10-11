"""X7 confirmation data: BIRD dev remainder (1,033 never-used questions), within-stack.
--phase gen (GPU, gold-free): greedy (frozen BIRD prompt) + frozen checker/critic + JIT generic repair + frozen
selector + 8 samples (T=0.7, top_p=0.95, per-item seeds), max_new_tokens 768.
--phase analyze (CPU): sample checks + result keys (gold-free) -> selections fixed -> BIRD evaluator (set equality,
30 s, checker-invalid = wrong, gold-failure items fixed incorrect) + evaluation guard -> per-item correctness dump."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DB = (jp.DATA / "bird/dev_20240627/dev_databases")
IN = (jp.OUTBOX / "x7/birddev_assets")
OUT = (jp.OUTBOX / "x7/birddev")
MODELS = {"llama31_8b": str(jp.MODELS / "llama31_8b"), "mistral7b": str(jp.MODELS / "mistral7b"),
          "qwen3_8b": str(jp.MODELS / "qwen3_8b"), "qwen3_14b": str(jp.MODELS / "qwen3_14b")}
NS = (1, 2, 3, 5, 8)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wjl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as h:
        for r in rows: h.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _check(task):
    import sql_tools
    from jit_repair import spider_feedback as sf, bird_minidev_feedback as bf
    from jit_repair.spider_critic_feedback import critic_score, lint_question_sql
    row, raw, want_key = task
    sql = sf.extract_sql(raw); db = bf.locate_database(DB, str(row["db_id"]))
    ex = sf.execute_readonly(db, sql, 30.0)
    issues = lint_question_sql(bf.critic_text(str(row["question"]), str(row["evidence"])), sql) if ex.valid else []
    out = {"sql": sql, "execution_valid": ex.valid, "error": ex.error, "error_class": ex.error_class,
           "critic_issues": issues, "critic_score": critic_score(issues), "needs_repair": (not ex.valid) or bool(issues)}
    if want_key:
        st, key = sql_tools.result_key(db, sql, 30.0) if ex.valid else ("invalid", None)
        out["key_status"], out["result_key"] = st, key
    return out


def gen(model):
    from gen_vllm import Engine
    from jit_repair import bird_minidev_feedback as bf
    from jit_repair.merge_spider_critic_retries import choose
    pool = mp.get_context("spawn").Pool(48)
    safe = jl(IN / "safe.jsonl"); by = {r["case_id"]: r for r in safe}; d = OUT / model
    prompts = [{"id": r["case_id"], "system": bf.BIRD_SQL_SYSTEM_PROMPT,
                "user": bf.render_initial_user(str(r["question"]), str(r["evidence"]), str(r["schema"]))} for r in safe]
    eng = Engine(MODELS[model], model, max_model_len=12288)
    if not (d / "final.jsonl").exists():
        g = eng.run(prompts, str(d / "gen_initial.jsonl"), temperature=0.0, max_tokens=768, tag=f"x7/{model}/initial")
        chk = pool.map(_check, [(by[p["id"]], g[p["id"]]["samples"][0]["text"], False) for p in prompts], chunksize=8)
        initial = []
        for p, c in zip(prompts, chk):
            s = g[p["id"]]["samples"][0]
            initial.append({"case_id": p["id"], "raw_output": s["text"], **c, "input_tokens": g[p["id"]]["input_tokens"],
                            "output_tokens": s.get("output_tokens_frozen_convention", s["output_tokens"]), "attempt": 1})
        wjl(d / "initial.jsonl", initial)
        flagged = [r for r in initial if r["needs_repair"]]
        rows = [{"id": r["case_id"], "system": bf.BIRD_SQL_SYSTEM_PROMPT,
                 "user": bf.render_repair_user(str(by[r["case_id"]]["question"]), str(by[r["case_id"]]["evidence"]),
                                               str(by[r["case_id"]]["schema"]), r["sql"], bf.render_generic_feedback())} for r in flagged]
        gr = eng.run(rows, str(d / "gen_repair_1.jsonl"), temperature=0.0, max_tokens=768, tag=f"x7/{model}/repair") if rows else {}
        chk = pool.map(_check, [(by[x["id"]], gr[x["id"]]["samples"][0]["text"], False) for x in rows], chunksize=8)
        reps = {}
        for x, c in zip(rows, chk):
            s = gr[x["id"]]["samples"][0]
            reps[x["id"]] = {"case_id": x["id"], "raw_output": s["text"], **c, "input_tokens": gr[x["id"]]["input_tokens"],
                             "output_tokens": s.get("output_tokens_frozen_convention", s["output_tokens"]), "attempt": 2}
        wjl(d / "repair_1.jsonl", list(reps.values()))
        final = []
        for r in initial:
            sel, reason = (choose(r, reps[r["case_id"]]) if r["case_id"] in reps else (r, "unflagged_previous_immutable"))
            sel = dict(sel); sel["selection_reason"] = reason
            sel["jit_calls"] = 1 + int(r["case_id"] in reps)
            sel["jit_output_tokens"] = r["output_tokens"] + (reps[r["case_id"]]["output_tokens"] if r["case_id"] in reps else 0)
            final.append(sel)
        wjl(d / "final.jsonl", final)
    eng.run(prompts, str(d / "gen_samples.jsonl"), n=8, temperature=0.7, top_p=0.95, max_tokens=768, seed=20261011,
            tag=f"x7/{model}/samples")
    (d / "GEN_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    pool.close()


def _score(task):
    import sql_tools
    from jit_repair import bird_minidev_feedback as bf
    cid, db_id, sql, gold_sql, valid, gold_ok = task
    if not gold_ok:
        return 0, "gold_fixed_incorrect"
    return sql_tools.score_bird(str(bf.locate_database(DB, db_id)), sql, gold_sql, valid, 30.0)


def analyze(models):
    import sql_tools
    safe = jl(IN / "safe.jsonl"); by = {r["case_id"]: r for r in safe}; ids = [r["case_id"] for r in safe]
    gold = {r["case_id"]: r for r in jl(IN / "gold.jsonl")}
    pool = mp.get_context("spawn").Pool(48)
    for model in models:
        d = OUT / model
        gs = {r["id"]: r for r in jl(d / "gen_samples.jsonl")}
        scp = d / "samples_checked.jsonl"
        if not scp.exists():
            res = pool.map(_check, [(by[c], s["text"], True) for c in ids for s in gs[c]["samples"]], chunksize=16)
            rows, k = [], 0
            for c in ids:
                ss = []
                for s in gs[c]["samples"]:
                    x = res[k]; k += 1
                    ss.append({"sql": x["sql"], "execution_valid": x["execution_valid"], "error_class": x["error_class"],
                               "critic_score": x["critic_score"], "key_status": x["key_status"], "result_key": x["result_key"],
                               "output_tokens": s.get("output_tokens_frozen_convention", s["output_tokens"])})
                rows.append({"case_id": c, "samples": ss})
            wjl(scp, rows)
        checked = {r["case_id"]: r["samples"] for r in jl(scp)}
        ini = {r["case_id"]: r for r in jl(d / "initial.jsonl")}; fin = {r["case_id"]: r for r in jl(d / "final.jsonl")}
        sel = {}
        for c in ids:
            g = {"sql": ini[c]["sql"], "execution_valid": ini[c]["execution_valid"]}
            sel[f"{c}|greedy"] = [ini[c]["sql"], ini[c]["execution_valid"], 1, ini[c]["output_tokens"]]
            sel[f"{c}|jit"] = [fin[c]["sql"], fin[c]["execution_valid"], fin[c]["jit_calls"], fin[c]["jit_output_tokens"]]
            for N in NS:
                ch, idx, calls = sql_tools.exec_vote(checked[c], N, g)
                tok = sum(s["output_tokens"] for s in checked[c][:N]) + (ini[c]["output_tokens"] if idx == "greedy" else 0)
                sel[f"{c}|vote@{N}"] = [ch["sql"], bool(ch["execution_valid"]), calls, tok,
                                        (checked[c][idx]["critic_score"] if idx != "greedy" else ini[c]["critic_score"])]
        json.dump(sel, open(d / "selections_gold_free.json", "w"))
        uniq = sorted({(k.split("|")[0], v[0], bool(v[1])) for k, v in sel.items()})
        sc = pool.map(_score, [(c, by[c]["db_id"], s, gold[c]["sql"], v, gold[c]["gold_executes"]) for c, s, v in uniq], chunksize=8)
        score = {k: v for k, v in zip(uniq, sc)}
        methods = ["greedy", "jit"] + [f"vote@{N}" for N in NS]
        with open(d / "per_item.jsonl", "w") as h:
            for c in ids:
                h.write(json.dumps({"case_id": c, **{m: score[(c, sel[f'{c}|{m}'][0], bool(sel[f'{c}|{m}'][1]))][0] for m in methods}}) + "\n")
        errs = [e for _, e in sc if e]
        n_exec = sum(len(checked[c]) for c in ids) + 2 * len(ids)
        to = (sum(s["error_class"] == "execution_timeout" for c in ids for s in checked[c]) + sum(s["key_status"] == "timeout" for c in ids for s in checked[c])
              + sum(1 for e in errs if ("interrupt" in e.lower() or "timeout" in e.lower() or e.startswith("guard_"))))
        json.dump({"model": model, "n": len(ids), "executions": n_exec, "timeouts": to, "timeout_rate": to / n_exec,
                   "evaluator_errors": len(errs), "flag_rate": sum(ini[c]["needs_repair"] for c in ids) / len(ids)},
                  open(d / "run_summary.json", "w"), indent=1)
        print("X7 analyzed", model, "timeout_rate", round(to / n_exec, 5), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True); ap.add_argument("--model")
    ap.add_argument("--models", default="llama31_8b,mistral7b,qwen3_8b")
    a = ap.parse_args()
    gen(a.model) if a.phase == "gen" else analyze(a.models.split(","))
