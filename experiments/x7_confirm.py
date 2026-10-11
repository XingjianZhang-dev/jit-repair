"""X7 confirmation analysis (pre-registered before any generation; Supplement S5).
P1: pooled over the evaluable models (items stacked, bootstrap 10,000, seed 20261011, resampling question ids so a
question's outputs from all models stay together), 97.5% CI of VER-G - RAND_U has lower bound > 0 AND every model's
point estimate > 0.  P2: same for VER-G - CONF_sum_U, lower bound > 0 AND every model's point estimate >= 0.  N = 3.
A model is NOT EVALUABLE if execution timeouts exceed 1% of its executions. Synthetic tests must pass first.
Everything else here is descriptive. CPU only; gold enters only through per-item correctness."""
from __future__ import annotations
import csv, json, os, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x7_alloc as xa  # noqa
from x7_confirm_load import load_model, MODELS, X7, REC, jl  # noqa
from jit_repair.merge_spider_critic_retries import choose  # noqa
CONF = ["CONF_mean_U", "CONF_mean_G", "CONF_min_U", "CONF_min_G", "CONF_sum_U", "CONF_sum_G"]


def synthetic_tests():
    rnd = random.Random(7); out = {}
    # T1: accept() == frozen choose() on random records
    bad = 0
    for _ in range(5000):
        g = {"sql": rnd.choice(["a", "b", "c"]), "execution_valid": rnd.random() < .7, "critic_score": rnd.randint(0, 3)}
        w = {"sql": rnd.choice(["a", "b", "c"]), "execution_valid": rnd.random() < .7, "critic_score": rnd.randint(0, 3)}
        took = choose(g, w)[0] is w
        mine = xa.accept(g, w["sql"], w["execution_valid"], w["critic_score"])
        if w["sql"] == g["sql"]:
            continue  # identical SQL -> identical outcome either way
        bad += int(took != mine)
    out["T1_accept_equals_frozen_choose"] = bad == 0
    # T2: RAND expectation == Monte-Carlo mean over random size-k subsets
    n = 400; ids = [f"s{i}" for i in range(n)]
    ini = {c: {"needs_repair": rnd.random() < .3, "sql": "g", "execution_valid": rnd.random() < .8, "critic_score": rnd.randint(0, 2),
               "output_tokens": rnd.randint(5, 50)} for c in ids}
    pi = {c: {"greedy": int(rnd.random() < .5), "vote@3": int(rnd.random() < .55)} for c in ids}
    win = {3: {c: (rnd.choice(["g", "w"]), rnd.random() < .9, rnd.randint(0, 2)) for c in ids}}
    lp = {c: {"mean_lp": -rnd.random(), "min_lp": -3 * rnd.random(), "sum_lp": -20 * rnd.random()} for c in ids}
    ours, comps, f, k, g = xa.vectors(ids, ini, win, pi, lp, {c: rnd.randint(10, 99) for c in ids}, 3)
    w = {c: pi[c]["vote@3"] for c in ids}; acc = {c: xa.accept(ini[c], *win[3][c]) for c in ids}
    mc_u, mc_g = [], []
    for _ in range(20000):
        T = set(rnd.sample(ids, k))
        mc_u.append(np.mean([w[c] if c in T else g[c] for c in ids])); mc_g.append(np.mean([w[c] if (c in T and acc[c]) else g[c] for c in ids]))
    out["T2_rand_expectation"] = bool(abs(np.mean(mc_u) - np.mean(list(comps["RAND_U"].values()))) < 2e-3 and abs(np.mean(mc_g) - np.mean(list(comps["RAND_G"].values()))) < 2e-3)
    # T3: gate == oracle -> VER_U == ORACLE_U; topk returns exactly k
    ini2 = {c: dict(ini[c], needs_repair=(pi[c]["greedy"] == 0)) for c in ids}
    _, comps2, _, k2, _ = xa.vectors(ids, ini2, win, pi, lp, {c: 1 for c in ids}, 3)
    out["T3_gate_equals_oracle"] = comps2["VER_U"] == comps2["ORACLE_U"] and len(xa.topk(ids, {c: 0 for c in ids}, k2)) == k2
    # T4: clustered bootstrap on stacked data with equal cluster sizes: mean exact, 97.5% CI contains 95% CI
    d = [rnd.gauss(0.02, 0.3) for _ in range(800)]; cl = [i // 4 for i in range(800)]
    m1, lo1, hi1 = xa.boot(d, cl, 0.95); m2, lo2, hi2 = xa.boot(d, cl, 0.975)
    out["T4_cluster_bootstrap"] = abs(m1 - np.mean(d)) < 1e-12 and lo2 <= lo1 <= m1 <= hi1 <= hi2
    # T5: null coverage of the 97.5% clustered CI on 200 synthetic pools (true mean 0, 300 questions x 4 models with a
    # shared per-question effect); must be >= 95%
    cov = 0; rng = np.random.default_rng(3)
    for r in range(200):
        q = rng.normal(0, 0.3, 300); dd = (q[:, None] + rng.normal(0, .2, (300, 4))).ravel()
        m, lo, hi = xa.boot(dd, np.repeat(np.arange(300), 4), 0.975)
        cov += int(lo <= 0 <= hi)
    out["T5_null_coverage"] = bool(cov / 200 >= 0.95)
    out["T5_null_coverage_rate"] = cov / 200
    return out


def diversity(m):
    """Gold-free: mean #distinct executable result keys among the first 3 / 8 samples, and share of items whose
    first 3 executable samples all agree."""
    rows = jl(X7 / "birddev" / m / "samples_checked.jsonl"); d3, d8, agree3 = [], [], []
    for r in rows:
        k3 = [s["result_key"] for s in r["samples"][:3] if s["execution_valid"] and s["key_status"] == "ok"]
        k8 = [s["result_key"] for s in r["samples"] if s["execution_valid"] and s["key_status"] == "ok"]
        d3.append(len(set(map(json.dumps, k3)))); d8.append(len(set(map(json.dumps, k8))))
        agree3.append(int(len(k3) == 3 and len(set(map(json.dumps, k3))) == 1))
    return float(np.mean(d3)), float(np.mean(d8)), float(np.mean(agree3))


def main():
    tests = synthetic_tests(); print("synthetic tests", tests, flush=True)
    assert all(v for k, v in tests.items() if not k.endswith("_rate")), "synthetic tests failed"
    safe = {r["case_id"]: r for r in jl(X7 / "birddev_assets/safe.jsonl")}
    summ = {m: json.load(open(X7 / "birddev" / m / "run_summary.json")) for m in MODELS}
    evaluable = [m for m in MODELS if summ[m]["timeout_rate"] <= 0.01]
    cells = {m: load_model(m) for m in MODELS}
    out = {"preregistration_commit": "53c6ff0", "n_questions": len(safe), "models": list(MODELS), "run_summary": summ,
           "evaluable": evaluable, "synthetic_tests": tests, "N_primary": 3}
    per = {}
    for N in (2, 3, 5):
        vec = {}
        for m in MODELS:
            name, ids, ini, win, pi, lp, qlen, _ = cells[m]
            ours, comps, f, k, g = xa.vectors(ids, ini, win, pi, lp, qlen, N)
            vec[m] = (ids, ours, comps, f, g)
        per[N] = vec
    # ---- primary: P1, P2 at N = 3
    def pooled(N, comp, models, level):
        d, cl = [], []
        for m in models:
            ids, ours, comps, f, g = per[N][m]
            d += [ours[c] - comps[comp][c] for c in ids]; cl += ids
        return xa.boot(d, cl, level)
    res = {}
    for P, comp, rule in (("P1", "RAND_U", ">"), ("P2", "CONF_sum_U", ">=")):
        mm, lo, hi = pooled(3, comp, evaluable, 0.975)
        pts = {m: float(np.mean([per[3][m][1][c] - per[3][m][2][comp][c] for c in per[3][m][0]])) for m in evaluable}
        ok_models = all((v > 0) if rule == ">" else (v >= 0) for v in pts.values())
        verdict = "CONFIRMED" if (lo > 0 and ok_models and len(evaluable) > 0) else "NOT CONFIRMED"
        res[P] = {"comparator": comp, "n_items": len(safe) * len(evaluable), "n_questions": len(safe), "models": evaluable,
                  "pooled_diff": mm, "ci97_5": [lo, hi], "per_model_point": pts, "per_model_rule": rule, "verdict": verdict}
        print(P, verdict, f"pooled {100*mm:+.2f} [{100*lo:+.2f},{100*hi:+.2f}]", {m: round(100 * v, 2) for m, v in pts.items()}, flush=True)
    out["primary"] = res
    # ---- descriptive: per model, every comparator, 95% CI, N = 2, 3, 5
    rows = []
    for N in (2, 3, 5):
        for m in MODELS:
            ids, ours, comps, f, g = per[N][m]
            base = {"model": m, "N": N, "n": len(ids), "trigger_rate": f, "greedy": float(np.mean(list(g.values()))),
                    "VER_G": float(np.mean(list(ours.values())))}
            for comp, other in comps.items():
                mm, lo, hi = xa.boot([ours[c] - other[c] for c in ids])
                rows.append({**base, "comparator": comp, "comparator_acc": float(np.mean(list(other.values()))), "VER_G_minus": mm, "ci_lo": lo, "ci_hi": hi})
        for comp in ["RAND_U", "RAND_G", "CONF_sum_U", "CONF_mean_U", "CONF_min_U", "CONF_min_G", "LEN_sql_U", "LEN_q_U", "VER_U", "ORACLE_G", "ORACLE_U"]:
            mm, lo, hi = pooled(N, comp, evaluable, 0.95)
            rows.append({"model": "POOLED", "N": N, "n": len(safe) * len(evaluable), "trigger_rate": float("nan"), "greedy": float("nan"),
                         "VER_G": float("nan"), "comparator": comp, "comparator_acc": float("nan"), "VER_G_minus": mm, "ci_lo": lo, "ci_hi": hi})
    with open(REC / "tables/X7_confirmation_allocation.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # ---- descriptive: best-of-six confidence, spend vs JIT and ungated vote, diversity, difficulty
    desc = {}
    for m in MODELS:
        ids, ours, comps, f, g = per[3][m]
        sel = json.load(open(X7 / "birddev" / m / "selections_gold_free.json")); pi = cells[m][4]
        best = max(CONF, key=lambda k: np.mean(list(comps[k].values())))
        b_m, b_lo, b_hi = xa.boot([ours[c] - comps[best][c] for c in ids])
        jit = {c: pi[c]["jit"] for c in ids}
        j_m, j_lo, j_hi = xa.boot([ours[c] - jit[c] for c in ids])
        sc = {r["case_id"]: r["samples"] for r in jl(X7 / "birddev" / m / "samples_checked.jsonl")}
        ini = cells[m][2]
        ver_calls = {N: float(np.mean([1 + (N if ini[c]["needs_repair"] else 0) for c in ids])) for N in (2, 3, 5)}
        ver_tok = {N: float(np.mean([ini[c]["output_tokens"] + (sum(s["output_tokens"] for s in sc[c][:N]) if ini[c]["needs_repair"] else 0) for c in ids])) for N in (2, 3, 5)}
        vote = {N: {"acc": float(np.mean([pi[c][f"vote@{N}"] for c in ids])), "calls": float(np.mean([sel[f"{c}|vote@{N}"][2] for c in ids])),
                    "out_tokens": float(np.mean([sel[f"{c}|vote@{N}"][3] for c in ids]))} for N in (1, 2, 3, 5, 8)}
        verg = {N: {"acc": float(np.mean(list(per[N][m][1].values()))), "calls": ver_calls[N], "out_tokens": ver_tok[N]} for N in (2, 3, 5)}
        jitd = {"acc": float(np.mean(list(jit.values()))), "calls": float(np.mean([sel[f"{c}|jit"][2] for c in ids])),
                "out_tokens": float(np.mean([sel[f"{c}|jit"][3] for c in ids]))}
        # ungated vote at matched calls: linear interpolation of vote@N accuracy at VER-G@3's mean calls
        xs = [vote[N]["calls"] for N in (1, 2, 3, 5, 8)]; ys = [vote[N]["acc"] for N in (1, 2, 3, 5, 8)]
        matched = float(np.interp(ver_calls[3], xs, ys))
        d3, d8, agree3 = diversity(m)
        diff_by = {}
        for lev in ("simple", "moderate", "challenging"):
            sub = [c for c in ids if safe[c]["difficulty"] == lev]
            mm, lo, hi = xa.boot([ours[c] - comps["RAND_U"][c] for c in sub])
            diff_by[lev] = {"n": len(sub), "VER_G_minus_RAND_U": mm, "ci95": [lo, hi]}
        prec = float(np.mean([1 - g[c] for c in ids if ini[c]["needs_repair"]])) if any(ini[c]["needs_repair"] for c in ids) else float("nan")
        base_err = 1 - float(np.mean(list(g.values())))
        desc[m] = {"trigger_rate": f, "greedy": float(np.mean(list(g.values()))), "flag_precision_greedy_wrong": prec, "base_error": base_err,
                   "best_of_six_conf": best, "best_conf_acc": float(np.mean(list(comps[best].values()))), "VER_G_minus_best_conf": [b_m, b_lo, b_hi],
                   "VER_G_minus_JIT": [j_m, j_lo, j_hi], "JIT": jitd, "VER_G": verg, "ungated_vote": vote,
                   "ungated_vote_acc_at_VER_G3_calls_interp": matched, "VER_G3_minus_matched_ungated": verg[3]["acc"] - matched,
                   "diversity_distinct_keys_first3": d3, "diversity_distinct_keys_8": d8, "all3_agree_rate": agree3, "by_difficulty_vs_RAND_U": diff_by,
                   "timeout_rate": summ[m]["timeout_rate"]}
        print(m, f"f={f:.3f} prec={prec:.3f} greedy={100*desc[m]['greedy']:.2f} VER-G3={100*verg[3]['acc']:.2f} JIT={100*jitd['acc']:.2f} "
              f"bestconf={best} Δ={100*b_m:+.2f}[{100*b_lo:+.2f},{100*b_hi:+.2f}] vsJIT={100*j_m:+.2f} matchedvote={100*matched:.2f} d3={d3:.2f} agree3={agree3:.3f}", flush=True)
    out["descriptive"] = desc
    # pooled difficulty split vs RAND_U (descriptive)
    pdiff = {}
    for lev in ("simple", "moderate", "challenging"):
        d, cl = [], []
        for m in evaluable:
            ids, ours, comps, f, g = per[3][m]
            for c in ids:
                if safe[c]["difficulty"] == lev:
                    d.append(ours[c] - comps["RAND_U"][c]); cl.append(c)
        mm, lo, hi = xa.boot(d, cl)
        pdiff[lev] = {"n_items": len(d), "n_questions": len(set(cl)), "VER_G_minus_RAND_U": mm, "ci95": [lo, hi]}
    out["pooled_by_difficulty_vs_RAND_U"] = pdiff
    json.dump(out, open(REC / "tables/X7_confirmation_pooled.json", "w"), indent=1)
    print("written", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        print(synthetic_tests())
    else:
        main()
