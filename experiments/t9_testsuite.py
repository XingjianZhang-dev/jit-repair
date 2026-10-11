"""T9: semantic SQL evaluation with the Zhong et al. (2020) test suite (distilled database variants).
Spider test: test-suite DBs cover Spider dev only -> check coverage. Dr.Spider NLQ+SQL axes reuse the original
Spider-dev databases -> re-score the stored direct and JIT-Repair (full) predictions on the test-suite variants."""
from jit_repair import paths as jp  # noqa: E402
import json, multiprocessing as mp, os, random, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TS = str(jp.DATA / "dl/test-suite-sql-eval")
TSDB = (jp.DATA / "testsuite/database")
OUT = (jp.OUTBOX / "t9"); OUT.mkdir(parents=True, exist_ok=True)


def _score(t):
    i, db, p, g, valid = t
    if not valid:
        return 0, None
    random.seed(20260822 + i)
    if TS not in sys.path:
        sys.path.insert(0, TS)
    import sql_tools
    gs = sql_tools.guard(TSDB / db / f"{db}.sqlite", p, 60.0)   # paper's Dr.Spider per-query budget = 60 s
    if gs in ("timeout", "too_many_rows"):
        return 0, f"guard_{gs}"
    try:
        from exec_eval import eval_exec_match
        return int(eval_exec_match(db=str(TSDB / db / f"{db}.sqlite"), p_str=p, g_str=g, plug_value=False,
                                   keep_distinct=False, progress_bar_for_each_datapoint=False)), None
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def main():
    import paired_stats as st
    test_dbs = {p.name for p in (jp.DATA / "spider/spider_data/test_database").iterdir() if p.is_dir()}
    ts_dbs = {p.name for p in TSDB.iterdir() if p.is_dir()}
    cov = {"spider_test_db_ids": len(test_dbs), "testsuite_db_ids": len(ts_dbs),
           "spider_test_ids_covered_by_testsuite": sorted(test_dbs & ts_dbs)}
    # Spider-test ids that ALSO exist in Spider dev would be covered only if the test split reused them
    json.dump(cov, open(OUT / "coverage.json", "w"), indent=1); print(cov)
    pool = mp.get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 48)
    summary = {"coverage": cov}
    for m in ("llama31_8b", "mistral7b"):
        rows = [json.loads(l) for l in open(str(jp.INBOX / f"t9/{m}.jsonl"))]
        missing = sorted({r["db_id"] for r in rows if r["db_id"] not in ts_dbs})
        rows = [r for r in rows if r["db_id"] in ts_dbs]
        res = {}
        for arm, sk, vk in (("gold_control", "gold_sql", None), ("direct", "sql_direct", "valid_direct"), ("full", "sql_full", "valid_full")):
            tasks = [(i, r["db_id"], r[sk], r["gold_sql"], True if vk is None else bool(r[vk])) for i, r in enumerate(rows)]
            res[arm] = pool.map(_score, tasks, chunksize=16)
        per = []
        for i, r in enumerate(rows):
            gi = bool(r.get("gold_self_inconsistent"))
            per.append({"case_id": r["case_id"], "q_id": r["q_id"], "axis": r["axis"], "set_name": r["set_name"],
                        "gold_control": res["gold_control"][i][0],
                        "ts_direct": 0 if gi else res["direct"][i][0], "ts_full": 0 if gi else res["full"][i][0],
                        "paper_direct": int(r["paper_correct_direct"]), "paper_full": int(r["paper_correct_full"]),
                        "err_direct": res["direct"][i][1], "err_full": res["full"][i][1]})
        with open(OUT / f"{m}_per_item.jsonl", "w") as h:
            for p in per: h.write(json.dumps(p) + "\n")
        cl = {p["case_id"]: p["q_id"] for p in per}
        ts_d = {p["case_id"]: p["ts_direct"] for p in per}; ts_f = {p["case_id"]: p["ts_full"] for p in per}
        pa_d = {p["case_id"]: p["paper_direct"] for p in per}; pa_f = {p["case_id"]: p["paper_full"] for p in per}
        s = {"n": len(per), "clusters": len(set(cl.values())), "missing_db_ids": missing,
             "gold_control_pass_rate": sum(p["gold_control"] for p in per) / len(per),
             "testsuite_full_minus_direct": st.paired(ts_f, ts_d, seed=20261011, clusters=cl),
             "paper_single_db_full_minus_direct_same_items": st.paired(pa_f, pa_d, seed=20261011, clusters=cl)}
        for arm, ts, pa in (("direct", ts_d, pa_d), ("full", ts_f, pa_f)):
            s[f"{arm}_2x2_paper_vs_testsuite"] = {
                "both_correct": sum(pa[c] and ts[c] for c in pa), "paper_only": sum(pa[c] and not ts[c] for c in pa),
                "testsuite_only": sum((not pa[c]) and ts[c] for c in pa), "both_wrong": sum((not pa[c]) and (not ts[c]) for c in pa)}
        s["evaluator_errors"] = {a: sum(e is not None for _, e in res[a]) for a in res}
        summary[m] = s
        print(m, json.dumps({k: v for k, v in s.items() if "minus" not in k}, indent=0))
        print(m, "TS gain", s["testsuite_full_minus_direct"]["difference"], s["testsuite_full_minus_direct"]["ci95"],
              "paper gain same items", s["paper_single_db_full_minus_direct_same_items"]["difference"])
    json.dump(summary, open(OUT / "t9_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
