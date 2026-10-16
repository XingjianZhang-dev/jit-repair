"""Gated voting against three-sample self-consistency over the 11 SQL cells with stored vote samples (descriptive;
reads the stored per-question records only).

Per cell and pooled over the cells: accuracy gain over greedy decoding and correct greedy outputs broken, for
self-consistency (a three-sample vote on every question) and for JIT-Repair with the vote spend (gate, vote of N,
selector) at N = 3; and JIT-Repair at the largest stored vote size N in {3, 5, 8} whose measured model calls do not
exceed those of three-sample self-consistency in that cell. Pooled intervals: paired bootstrap, 10,000 resamples,
questions resampled as clusters (a benchmark question with all its model outputs; Dr.Spider: source question).
Uses the loaders, selector and bootstrap of experiments/x7_alloc.py unchanged. Writes analysis/equal_budget.csv."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import csv, json, os, sys
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXP))
import x7_alloc as xa  # noqa: E402
from x7_confirm_load import load_model, MODELS  # noqa: E402
OUT = jp.outputs("analysis")
REC = jp.RECORDS
NS = (3, 5, 8)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def selections(name):
    b, m = name.split("/")
    if b in ("spider", "bird"):
        return json.load(open(REC / f"results/c2/{b}/{m}/selections_gold_free.json")), 2
    if b == "drspider_subset":
        return json.load(open(REC / f"results/x3b/{m}/selections_gold_free.json")), 1
    return json.load(open(REC / f"results/x7/birddev/{m}/selections_gold_free.json")), 2


def question_text(name):
    """Question text the critic reads, per case (as in x7_alloc.load_existing)."""
    b = name.split("/")[0]
    if b == "spider":
        return {str(r["case_id"]): str(r["question"]) for r in jl(xa.E / "spider_generic_forward_canonical_v2_full/prepared/safe.jsonl")}
    if b == "bird":
        return {str(r["case_id"]): xa.critic_text(str(r["question"]), str(r["evidence"]))
                for r in jl(xa.E / "bird_minidev_generic_forward_assets_v2/safe.jsonl")}
    return {r["case_id"]: str(r["question"]) for r in jl(jp.RECORDS / "inputs/drspider/safe_subset.jsonl")}


def with_eight(cell):
    """Adds the vote-of-eight winner (sql, checker validity, critic score) to the cell's win dict."""
    name, ids, ini, win, pi, lp, qlen, clusters = cell
    if name.startswith("birddev_rest/"):
        return load_model(name.split("/")[1], Ns=(2, 3, 5, 8))
    sel, _ = selections(name); q = question_text(name); w8 = {}
    for c in ids:
        s = sel[f"{c}|vote@8"]
        if name.startswith("drspider_subset/"):
            v = ini[c]["execution_valid"] if s[0] == ini[c]["sql"] else True
        else:
            v = bool(s[1])
        w8[c] = (s[0], bool(v), xa.critic_score(xa.lint_question_sql(q[c], s[0])))
    return (name, ids, ini, {**win, 8: w8}, pi, lp, qlen, clusters)


def main() -> None:
    cells = [with_eight(c) for c in list(xa.load_existing()) + [load_model(m) for m in MODELS]]
    rows, pool = [], {k: [] for k in ("g", "sc", "j3", "jn", "cl", "c_sc", "c_j3", "c_jn")}
    for cell in cells:
        name, ids, ini, win, pi, lp, qlen, clusters = cell
        sel, ci = selections(name)
        G = np.array([1.0 if ini[c]["needs_repair"] else 0.0 for c in ids])
        g = np.array([float(pi[c]["greedy"]) for c in ids])
        sc = np.array([float(pi[c]["vote@3"]) for c in ids])
        c_sc = np.array([float(sel[f"{c}|vote@3"][ci]) for c in ids])
        ours = {}
        for N in NS:
            v = xa.vectors(ids, ini, win, pi, lp, qlen, N)[0]
            ours[N] = np.array([float(v[c]) for c in ids])
        calls = {N: 1.0 + N * G for N in NS}
        nstar = max(N for N in NS if calls[N].mean() <= c_sc.mean())
        bench = name.split("/")[0]
        cl = [f"{bench}:{clusters[c] if clusters else c}" for c in ids]
        rows.append({"cell": name, "n": len(ids), "rho": G.mean(), "N_star": nstar, "greedy": g.mean(), "sc3": sc.mean(),
                     "jit3": ours[3].mean(), "jitN": ours[nstar].mean(), "calls_sc3": c_sc.mean(), "calls_jit3": calls[3].mean(),
                     "calls_jitN": calls[nstar].mean(), "broken_sc3": int(np.sum((g == 1) & (sc == 0))),
                     "broken_jit3": int(np.sum((g == 1) & (ours[3] == 0))), "broken_jitN": int(np.sum((g == 1) & (ours[nstar] == 0))),
                     "fixed_sc3": int(np.sum((g == 0) & (sc == 1))), "fixed_jit3": int(np.sum((g == 0) & (ours[3] == 1))),
                     "fixed_jitN": int(np.sum((g == 0) & (ours[nstar] == 1))),
                     "diff_jitN_sc3": float((ours[nstar] - sc).mean())})
        for k, v in (("g", g), ("sc", sc), ("j3", ours[3]), ("jn", ours[nstar]), ("c_sc", c_sc), ("c_j3", calls[3]),
                     ("c_jn", calls[nstar])):
            pool[k] += list(v)
        pool["cl"] += cl
    P = {k: (np.array(v) if k != "cl" else v) for k, v in pool.items()}
    g, sc, j3, jn, cl = P["g"], P["sc"], P["j3"], P["jn"], P["cl"]
    br_sc = ((g == 1) & (sc == 0)).astype(float); br_j3 = ((g == 1) & (j3 == 0)).astype(float)
    br_jn = ((g == 1) & (jn == 0)).astype(float)
    stats = {
        "gain_sc3": xa.boot(sc - g, cl), "gain_jit3": xa.boot(j3 - g, cl), "gain_jitN": xa.boot(jn - g, cl),
        "diff_jit3_sc3": xa.boot(j3 - sc, cl), "diff_jitN_sc3": xa.boot(jn - sc, cl),
        "broken_sc3": xa.boot(br_sc, cl), "broken_jit3": xa.boot(br_j3, cl), "broken_jitN": xa.boot(br_jn, cl),
        "diff_broken_jit3_sc3": xa.boot(br_j3 - br_sc, cl), "diff_broken_jitN_sc3": xa.boot(br_jn - br_sc, cl),
    }
    n = len(g)
    for k, (m, lo, hi) in stats.items():
        rows.append({"cell": "POOLED", "n": n, "stat": k, "estimate": m, "ci_lo": lo, "ci_hi": hi})
    rows.append({"cell": "POOLED", "n": n, "stat": "calls_sc3", "estimate": float(P["c_sc"].mean())})
    rows.append({"cell": "POOLED", "n": n, "stat": "calls_jit3", "estimate": float(P["c_j3"].mean())})
    rows.append({"cell": "POOLED", "n": n, "stat": "calls_jitN", "estimate": float(P["c_jn"].mean())})
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(OUT / "equal_budget.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
