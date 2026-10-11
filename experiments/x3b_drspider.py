"""X3b confirmation on the frozen Dr.Spider ablation subset (1,556 items), within-stack.
--phase gen (GPU, gold-free): greedy + frozen checker/critic + JIT (generic repair + frozen selector) + 8 samples.
--phase analyze (CPU): sample checks/keys (gold-free) -> all selections fixed -> Dr.Spider evaluator -> stats."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import argparse, json, multiprocessing as mp, os, random, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = (jp.DATA / "drspider/data_official")
IN = (jp.INBOX / "drspider")
OUT = (jp.OUTBOX / "x3b")
TS = str(jp.DATA / "dl/test-suite-sql-eval")
MODELS = {"llama31_8b": str(jp.MODELS / "llama31_8b"), "mistral7b": str(jp.MODELS / "mistral7b"),
          "qwen3_8b": str(jp.MODELS / "qwen3_8b")}
NS = (1, 2, 3, 5, 8)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wjl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as h:
        for r in rows: h.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _check(task):
    from jit_repair import spider_feedback as sf
    from jit_repair.drspider_oneuse import locate_drspider_database
    from jit_repair.spider_critic_feedback import critic_score, lint_question_sql
    row, raw, want_key = task
    sql = sf.extract_sql(raw)
    db = locate_drspider_database(DATA, str(row["set_name"]), str(row["db_id"]))
    ex = sf.execute_readonly(db, sql, 30.0)
    issues = lint_question_sql(str(row["question"]), sql) if ex.valid else []
    out = {"sql": sql, "execution_valid": ex.valid, "error": ex.error, "error_class": ex.error_class,
           "critic_issues": issues, "critic_score": critic_score(issues), "needs_repair": (not ex.valid) or bool(issues)}
    if want_key:
        import sql_tools
        st, key = sql_tools.result_key(db, sql, 30.0) if ex.valid else ("invalid", None)
        out["key_status"], out["result_key"] = st, key
    return out


def gen(model):
    from gen_vllm import Engine
    from jit_repair import drspider_oneuse as do
    from jit_repair.merge_spider_critic_retries import choose
    pool = mp.get_context("spawn").Pool(48)
    safe = jl(IN / "safe_subset.jsonl"); by = {r["case_id"]: r for r in safe}
    d = OUT / model
    prompts = [{"id": r["case_id"], "system": do.DRSPIDER_SQL_SYSTEM_PROMPT,
                "user": do.render_initial_user(str(r["question"]), str(r["schema"]))} for r in safe]
    eng = Engine(MODELS[model], model, max_model_len=8192)
    if not (d / "final.jsonl").exists():
        g = eng.run(prompts, str(d / "gen_initial.jsonl"), temperature=0.0, max_tokens=768, tag=f"x3b/{model}/initial")
        chk = pool.map(_check, [(by[p["id"]], g[p["id"]]["samples"][0]["text"], False) for p in prompts], chunksize=8)
        initial = []
        for p, c in zip(prompts, chk):
            s = g[p["id"]]["samples"][0]
            initial.append({"case_id": p["id"], "raw_output": s["text"], **c, "input_tokens": g[p["id"]]["input_tokens"],
                            "output_tokens": s.get("output_tokens_frozen_convention", s["output_tokens"]), "attempt": 1})
        wjl(d / "initial.jsonl", initial)
        flagged = [r for r in initial if r["needs_repair"]]
        rows = [{"id": r["case_id"], "system": do.DRSPIDER_SQL_SYSTEM_PROMPT,
                 "user": do.render_generic_repair_user(str(by[r["case_id"]]["question"]), str(by[r["case_id"]]["schema"]), r["sql"])}
                for r in flagged]
        gr = eng.run(rows, str(d / "gen_repair_1.jsonl"), temperature=0.0, max_tokens=768, tag=f"x3b/{model}/repair")
        chk = pool.map(_check, [(by[x["id"]], gr[x["id"]]["samples"][0]["text"], False) for x in rows], chunksize=8)
        reps = {}
        for x, c in zip(rows, chk):
            s = gr[x["id"]]["samples"][0]
            reps[x["id"]] = {"case_id": x["id"], "raw_output": s["text"], **c, "input_tokens": gr[x["id"]]["input_tokens"],
                             "output_tokens": s.get("output_tokens_frozen_convention", s["output_tokens"]), "attempt": 2}
        wjl(d / "repair_1.jsonl", list(reps.values()))
        final = []
        for r in initial:
            if r["case_id"] in reps:
                sel, reason = choose(r, reps[r["case_id"]])
            else:
                sel, reason = r, "unflagged_previous_immutable"
            sel = dict(sel); sel["selection_reason"] = reason
            sel["jit_calls"] = 1 + int(r["case_id"] in reps)
            sel["jit_output_tokens"] = r["output_tokens"] + (reps[r["case_id"]]["output_tokens"] if r["case_id"] in reps else 0)
            final.append(sel)
        wjl(d / "final.jsonl", final)
    eng.run(prompts, str(d / "gen_samples.jsonl"), n=8, temperature=0.7, top_p=0.95, max_tokens=768, seed=20261011,
            tag=f"x3b/{model}/samples")
    (d / "GEN_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    pool.close()


_EV = {}
def _init(root):
    import importlib
    sys.path.insert(0, root)
    from jit_repair import evaluate_drspider_execution as ede
    ede._init_worker(root)
    _EV["ede"] = ede


def _score(task):
    """Frozen Dr.Spider evaluator + the evaluation guard (applied to every arm identically)."""
    import sql_tools
    from jit_repair.drspider_oneuse import locate_drspider_database
    idx, row, gold_sql, sql = task
    db = locate_drspider_database(DATA, str(row["set_name"]), str(row["db_id"]))
    g = sql_tools.guard(db, sql, 60.0) if sql.strip() else "ok"
    if g in ("timeout", "too_many_rows"):
        return 0, f"guard_{g}"
    suite = tuple(str(p) for p in sorted(db.parent.glob("*.sqlite")))
    _, correct, perr, eerr = _EV["ede"]._evaluate_one((idx, suite, sql, gold_sql, 60.0))
    return int(correct), eerr


def analyze(models=("llama31_8b", "mistral7b"), out_name="x3b_summary.json"):
    import sql_tools, paired_stats as st
    safe = jl(IN / "safe_subset.jsonl"); by = {r["case_id"]: r for r in safe}; ids = [r["case_id"] for r in safe]
    gold = {r["case_id"]: r["sql"] for r in jl(IN / "gold_subset.jsonl")}
    pool = mp.get_context("spawn").Pool(40, initializer=_init, initargs=(TS,))
    cpool = mp.get_context("spawn").Pool(40)
    per_model, pooled = {}, {"g": {}, "j": {}, "cl": {}}
    for model in models:
        d = OUT / model
        gen_s = {r["id"]: r for r in jl(d / "gen_samples.jsonl")}
        scp = d / "samples_checked.jsonl"
        if not scp.exists():
            tasks = [(by[c], s["text"], True) for c in ids for s in gen_s[c]["samples"]]
            res = cpool.map(_check, tasks, chunksize=16); k = 0; rows = []
            for c in ids:
                ss = []
                for s in gen_s[c]["samples"]:
                    ss.append({**{x: res[k][x] for x in ("sql", "execution_valid", "error_class", "key_status", "result_key")},
                               "output_tokens": s.get("output_tokens_frozen_convention", s["output_tokens"])}); k += 1
                rows.append({"case_id": c, "samples": ss})
            wjl(scp, rows)
        checked = {r["case_id"]: r["samples"] for r in jl(scp)}
        ini = {r["case_id"]: r for r in jl(d / "initial.jsonl")}; fin = {r["case_id"]: r for r in jl(d / "final.jsonl")}
        sel = {}
        for c in ids:
            g = {"sql": ini[c]["sql"], "execution_valid": ini[c]["execution_valid"]}
            flag = bool(ini[c]["needs_repair"])
            sel[(c, "greedy")] = (ini[c]["sql"], 1, ini[c]["output_tokens"])
            sel[(c, "jit")] = (fin[c]["sql"], fin[c]["jit_calls"], fin[c]["jit_output_tokens"])
            for N in NS:
                ch, idx, calls = sql_tools.exec_vote(checked[c], N, g)
                tok = sum(s["output_tokens"] for s in checked[c][:N]) + (ini[c]["output_tokens"] if idx == "greedy" else 0)
                sel[(c, f"vote@{N}")] = (ch["sql"], calls, tok)
                if flag:
                    sel[(c, f"gated_vote@{N}")] = (ch["sql"], 1 + N, ini[c]["output_tokens"] + sum(s["output_tokens"] for s in checked[c][:N]))
                else:
                    sel[(c, f"gated_vote@{N}")] = (ini[c]["sql"], 1, ini[c]["output_tokens"])
        json.dump({f"{c}|{m}": v for (c, m), v in sel.items()}, open(d / "selections_gold_free.json", "w"))
        uniq = sorted({(c, v[0]) for (c, m), v in sel.items()})
        pos = {c: i for i, c in enumerate(ids)}
        sc = pool.map(_score, [(pos[c], by[c], gold[c], s) for c, s in uniq], chunksize=8)
        score = {k: v for k, v in zip(uniq, sc)}
        errs = [e for _, e in sc if e]
        methods = sorted({m for (_, m) in sel})
        correct = {m: {c: score[(c, sel[(c, m)][0])][0] for c in ids} for m in methods}
        n = len(ids); cl = {c: by[c]["q_id_spider_dev"] for c in ids}
        frontier = {m: {"accuracy": sum(correct[m].values()) / n, "mean_calls": sum(sel[(c, m)][1] for c in ids) / n,
                        "mean_generated_tokens": sum(sel[(c, m)][2] for c in ids) / n} for m in methods}
        stats = {f"gated_vote@{N}_minus_jit": st.paired(correct[f"gated_vote@{N}"], correct["jit"], seed=20261011, clusters=cl) for N in (2, 3, 5)}
        stats.update({f"gated_vote@{N}_minus_vote@{N}": st.paired(correct[f"gated_vote@{N}"], correct[f"vote@{N}"], seed=20261011, clusters=cl) for N in (2, 3, 5)})
        stats["jit_minus_vote@2"] = st.paired(correct["jit"], correct["vote@2"], seed=20261011, clusters=cl)
        stats["jit_minus_greedy"] = st.paired(correct["jit"], correct["greedy"], seed=20261011, clusters=cl)
        n_exec = sum(len(checked[c]) for c in ids) + 2 * n
        timeouts = (sum(s["error_class"] == "execution_timeout" for c in ids for s in checked[c]) +
                    sum(s["key_status"] == "timeout" for c in ids for s in checked[c]) +
                    sum(1 for e in errs if ("timeout" in e.lower() or "interrupt" in e.lower() or e.startswith("guard_"))))
        per_model[model] = {"n": n, "clusters": len(set(cl.values())), "flag_rate": sum(ini[c]["needs_repair"] for c in ids) / n,
                            "frontier": frontier, "stats": stats, "timeout_rate": timeouts / n_exec, "evaluator_errors": len(errs),
                            "evaluator_error_examples": errs[:5]}
        for c in ids:
            key = f"{model}|{c}"
            pooled["g"][key] = correct["gated_vote@3"][c]; pooled["j"][key] = correct["jit"][c]; pooled["cl"][key] = f"{model}|{cl[c]}"
        print(model, json.dumps({m: round(v["accuracy"] * 100, 2) for m, v in frontier.items()}), flush=True)
    pooled_res = st.paired(pooled["g"], pooled["j"], seed=20261011, clusters=pooled["cl"])
    a_pts = all(per_model[m]["stats"]["gated_vote@3_minus_jit"]["difference"] >= 0 for m in per_model)
    a_ci = pooled_res["ci95"][0] > 0
    b = all(per_model[m]["stats"]["gated_vote@3_minus_vote@3"]["difference"] >= -0.01 for m in per_model)
    evaluable = all(per_model[m]["timeout_rate"] <= 0.01 for m in per_model)
    verdict = ("CONFIRMED" if (a_pts and a_ci and b) else "NOT CONFIRMED") if evaluable else "NOT EVALUABLE"
    out = {"per_model": per_model, "pooled_gated_vote@3_minus_jit": pooled_res, "a_points": a_pts, "a_pooled_ci": a_ci,
           "b_within_1pt_of_vote3": b, "evaluable": evaluable, "X3b_verdict": verdict}
    json.dump(out, open(OUT / out_name, "w"), indent=1)
    for m in per_model:
        s = per_model[m]["stats"]
        for k in ("gated_vote@3_minus_jit", "gated_vote@3_minus_vote@3", "jit_minus_vote@2", "jit_minus_greedy"):
            print(m, k, round(s[k]["difference"] * 100, 2), [round(x * 100, 2) for x in s[k]["ci95"]])
    print("pooled", round(pooled_res["difference"] * 100, 2), [round(x * 100, 2) for x in pooled_res["ci95"]], "X3b:", verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True); ap.add_argument("--model")
    ap.add_argument("--models", default="llama31_8b,mistral7b"); ap.add_argument("--out-name", default="x3b_summary.json")
    a = ap.parse_args()
    gen(a.model) if a.phase == "gen" else analyze(tuple(a.models.split(",")), a.out_name)
