"""C2 analysis for one cell (bench x model). Step 1 (gold-free): frozen checker + result key for every
sample. Step 2 (gold): paper evaluator on every distinct SQL. Step 3: selectors, costs, frontier, stats."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402

import argparse, json, multiprocessing as mp, os, random, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INBOX = jp.INBOX
DB = {"spider": (jp.DATA / "spider/spider_data/test_database"),
      "bird": (jp.DATA / "bird/dev_20240627/dev_databases")}
SAFE = {"spider": INBOX / "frozen/spider_llama31/prepared/safe.jsonl", "bird": INBOX / "frozen/bird_assets/safe.jsonl"}
TS = str(jp.DATA / "dl/test-suite-sql-eval")
NS = (1, 2, 3, 5, 8)
KNOWN_GOLD_TIMEOUT = {"bird-mini-dev-701"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _gf(task):  # gold-free: checker + result key
    import sql_tools
    from jit_repair import spider_feedback as sf, bird_minidev_feedback as bf
    bench, row, raw = task
    c = sql_tools.check(bench, row, raw, DB[bench])
    out = {"sql": c["sql"], "execution_valid": c["execution_valid"], "error_class": c["error_class"]}
    if c["execution_valid"]:
        loc = sf.locate_database if bench == "spider" else bf.locate_database
        st, key = sql_tools.result_key(loc(DB[bench], str(row["db_id"])), c["sql"], 10.0 if bench == "spider" else 30.0)
        out["key_status"], out["result_key"] = st, key
    else:
        out["key_status"], out["result_key"] = "invalid", None
    return out


def _score(task):
    import sql_tools
    from jit_repair import bird_minidev_feedback as bf
    bench, idx, cid, db_id, sql, gold, valid = task
    random.seed(20260822 + idx)
    if bench == "spider":
        return sql_tools.score_spider(TS, str(DB["spider"]), db_id, sql, gold, valid)
    if cid in KNOWN_GOLD_TIMEOUT:
        return 0, None
    return sql_tools.score_bird(str(bf.locate_database(DB["bird"], db_id)), sql, gold, valid, 30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    import sql_tools, paired_stats
    d = (jp.OUTBOX / "c2") / a.bench / a.model
    safe = jl(SAFE[a.bench]); by = {str(r["case_id"]): r for r in safe}
    ids = [str(r["case_id"]) for r in safe]
    pool = mp.get_context("spawn").Pool(a.workers)
    gen = {r["id"]: r for r in jl(d / "gen_samples.jsonl")}
    assert set(gen) == set(ids), "samples incomplete"
    # ---- step 1 gold-free
    sc_path = d / "samples_checked.jsonl"
    if not sc_path.exists():
        tasks = [(a.bench, by[c], s["text"]) for c in ids for s in gen[c]["samples"]]
        res = pool.map(_gf, tasks, chunksize=16)
        rows, k = [], 0
        for c in ids:
            ss = []
            for s in gen[c]["samples"]:
                r = res[k]; k += 1
                ss.append({**r, "output_tokens": s.get("output_tokens_frozen_convention", s["output_tokens"]),
                           "finish_reason": s["finish_reason"]})
            rows.append({"case_id": c, "samples": ss})
        with open(sc_path, "w") as h:
            for r in rows: h.write(json.dumps(r) + "\n")
    checked = {r["case_id"]: r["samples"] for r in jl(sc_path)}
    initial = {r["case_id"]: r for r in jl(d / "initial.jsonl")}
    final = {r["case_id"]: r for r in jl(d / "final.jsonl")}
    # ---- gold-free selection (complete before any scoring)
    sel = {}
    for c in ids:
        g = {"sql": initial[c]["sql"], "execution_valid": initial[c]["execution_valid"]}
        for N in NS:
            ch, idx, calls = sql_tools.exec_filter(checked[c], N, g)
            tok = sum(s["output_tokens"] for s in checked[c][:calls if idx != "greedy" else N]) + (initial[c]["output_tokens"] if idx == "greedy" else 0)
            sel[(c, f"filter@{N}")] = (ch["sql"], ch["execution_valid"], calls, tok)
            ch, idx, calls = sql_tools.exec_vote(checked[c], N, g)
            tok = sum(s["output_tokens"] for s in checked[c][:N]) + (initial[c]["output_tokens"] if idx == "greedy" else 0)
            sel[(c, f"vote@{N}")] = (ch["sql"], ch["execution_valid"], calls, tok)
        sel[(c, "greedy")] = (initial[c]["sql"], initial[c]["execution_valid"], 1, initial[c]["output_tokens"])
        sel[(c, "jit")] = (final[c]["sql"], final[c]["execution_valid"], final[c]["jit_calls"], final[c]["jit_output_tokens"])
    json.dump({f"{c}|{m}": v for (c, m), v in sel.items()}, open(d / "selections_gold_free.json", "w"))
    # ---- step 2 gold scoring of every distinct (case, sql, valid)
    if a.bench == "spider":
        gold = {ids[i]: l.rstrip("\n").rsplit("\t", 1)[0] for i, l in enumerate(open(INBOX / "frozen/spider_llama31/prepared/gold.sql", encoding="utf-8"))}
    else:
        gold = {str(r["case_id"]): r["sql"] for r in jl(INBOX / "frozen/bird_assets/gold.jsonl")}
    uniq = {}
    for c in ids:
        for s in checked[c]:
            uniq[(c, s["sql"], bool(s["execution_valid"]))] = None
        for m in ("greedy", "jit"):
            sq, v, _, _ = sel[(c, m)]; uniq[(c, sq, bool(v))] = None
    keys = list(uniq)
    t0 = time.time()
    pos = {c: i for i, c in enumerate(ids)}
    scores = pool.map(_score, [(a.bench, pos[c], c, by[c]["db_id"], sq, gold[c], v) for (c, sq, v) in keys], chunksize=8)
    for k_, s_ in zip(keys, scores): uniq[k_] = s_
    eval_errors = [e for _, e in scores if e is not None]
    # ---- step 3
    methods = ["greedy", "jit"] + [f"{s}@{N}" for s in ("filter", "vote") for N in NS]
    correct = {m: {} for m in methods}; cost = {m: [0, 0] for m in methods}
    for c in ids:
        for m in methods:
            sq, v, calls, tok = sel[(c, m)]
            correct[m][c] = uniq[(c, sq, bool(v))][0]
            cost[m][0] += calls; cost[m][1] += tok
    oracle = {N: {c: int(any(uniq[(c, s["sql"], bool(s["execution_valid"]))][0] for s in checked[c][:N])) for c in ids} for N in NS}
    sample_acc = sum(uniq[(c, s["sql"], bool(s["execution_valid"]))][0] for c in ids for s in checked[c]) / (8 * len(ids))
    n = len(ids)
    frontier = {m: {"accuracy": sum(correct[m].values()) / n, "mean_calls": cost[m][0] / n,
                    "mean_generated_tokens": cost[m][1] / n} for m in methods}
    stats = {f"jit_minus_{m}": paired_stats.paired(correct["jit"], correct[m], seed=20261011) for m in methods if m != "jit"}
    n_exec = sum(len(checked[c]) for c in ids) + 2 * n
    checker_timeouts = sum(s["error_class"] == "execution_timeout" for c in ids for s in checked[c]) + \
        sum(initial[c]["error_class"] == "execution_timeout" for c in ids)
    key_timeouts = sum(s["key_status"] == "timeout" for c in ids for s in checked[c])
    eval_timeouts = sum(("interrupt" in e.lower() or "timeout" in e.lower() or e.startswith("guard_")) for e in eval_errors)
    guard_counts = {g: sum(e == g for e in eval_errors) for g in ("guard_timeout", "guard_too_many_rows")}
    distinct_sql = sum(len({s["sql"] for s in checked[c]}) for c in ids) / n
    summary = {"bench": a.bench, "model": a.model, "n": n, "frontier": frontier,
               "pass_at_N_oracle_descriptive": {N: sum(v.values()) / n for N, v in oracle.items()},
               "mean_single_sample_accuracy": sample_acc, "mean_distinct_sql_per_question": distinct_sql,
               "stats": stats, "executions": n_exec, "checker_timeouts": checker_timeouts, "key_timeouts": key_timeouts,
               "evaluator_errors": len(eval_errors), "evaluator_timeouts": eval_timeouts, "guard_counts": guard_counts,
               "timeout_rate": (checker_timeouts + key_timeouts + eval_timeouts) / n_exec,
               "flagged": sum(initial[c]["needs_repair"] for c in ids),
               "selection_reasons": {}, "scoring_seconds": time.time() - t0}
    for c in ids:
        r = final[c]["selection_reason"]; summary["selection_reasons"][r] = summary["selection_reasons"].get(r, 0) + 1
    json.dump(summary, open(d / "c2_summary.json", "w"), indent=1)
    with open(d / "per_item.jsonl", "w") as h:
        for c in ids:
            h.write(json.dumps({"case_id": c, **{m: correct[m][c] for m in methods}, **{f"oracle@{N}": oracle[N][c] for N in NS}}) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "stats"}, indent=1))
    print("JIT-vote@2", stats["jit_minus_vote@2"])
    pool.close()


if __name__ == "__main__":
    main()
