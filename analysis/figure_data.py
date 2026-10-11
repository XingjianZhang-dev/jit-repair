"""Writes the plotted numbers of every data figure to outputs/figure_data/*.csv (reads the records only)."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import csv, json, os, sys
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXP))
import x7_alloc as xa  # noqa: E402
from x7_confirm_load import load_model, MODELS, X7  # noqa: E402
REC = xa.REC
D = jp.outputs("figure_data")
D.mkdir(parents=True, exist_ok=True)
SEED, B = 20261011, 10000
LAB = {"llama31_8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B", "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B"}
BENCH = {"spider": "Spider", "bird": "BIRD Mini-Dev", "drspider_subset": "Dr.Spider", "birddev_rest": "BIRD dev (held out)"}


def wcsv(name, rows):
    with open(D / name, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def boot_mean(x, clusters=None):
    x = np.asarray(x, float); rng = np.random.default_rng(SEED)
    if clusters is None:
        m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    else:
        cl = sorted(set(clusters)); pos = {c: j for j, c in enumerate(cl)}
        s = np.zeros(len(cl)); n = np.zeros(len(cl))
        for v, c in zip(x, clusters): s[pos[c]] += v; n[pos[c]] += 1
        idx = rng.integers(0, len(cl), (B, len(cl))); m = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.quantile(m, [0.025, 0.975]); return float(x.mean()), float(lo), float(hi)


def fig2():
    rows = list(csv.DictReader(open(REC / "tables/X7_confirmation_allocation.csv")))
    pooled = json.load(open(REC / "tables/X7_confirmation_pooled.json"))
    out = []
    for comp, P in (("RAND_U", "P1"), ("CONF_sum_U", "P2")):
        for m in list(MODELS) + ["POOLED"]:
            r = [x for x in rows if x["model"] == m and x["N"] == "3" and x["comparator"] == comp][0]
            rec = {"comparator": comp, "model": LAB.get(m, "Pooled (4 models)"), "n": r["n"], "diff": float(r["VER_G_minus"]),
                   "lo95": float(r["ci_lo"]), "hi95": float(r["ci_hi"]), "lo975": "", "hi975": "", "verdict": ""}
            if m == "POOLED":
                q = pooled["primary"][P]
                rec.update({"diff": q["pooled_diff"], "lo975": q["ci97_5"][0], "hi975": q["ci97_5"][1], "verdict": q["verdict"]})
            out.append(rec)
    wcsv("fig2_x7_confirmation.csv", out)


def fig4():
    out = []
    cells = list(xa.load_existing()) + [load_model(m) for m in MODELS]
    for name, ids, ini, win, pi, lp, qlen, clusters in cells:
        b, m = name.split("/")
        G = np.array([bool(ini[c]["needs_repair"]) for c in ids])
        DU = np.array([float(pi[c]["vote@3"]) - float(pi[c]["greedy"]) for c in ids])
        cl = np.array([clusters[c] for c in ids]) if clusters else None
        rec = {"bench": BENCH[b], "model": LAB[m], "n": len(ids), "rho": float(G.mean())}
        for tag, mask in (("flagged", G), ("passed", ~G)):
            mm, lo, hi = boot_mean(DU[mask], None if cl is None else list(cl[mask]))
            rec.update({f"{tag}_n": int(mask.sum()), f"{tag}_gain": mm, f"{tag}_lo": lo, f"{tag}_hi": hi,
                        f"{tag}_fixed": int(np.sum(mask & (DU > 0))), f"{tag}_broken": int(np.sum(mask & (DU < 0)))})
        out.append(rec)
    wcsv("fig4_vote_gain_flagged_vs_passed.csv", out)


def fig5():
    s = json.load(open(REC / "results/x3b/x3b_summary.json"))["per_model"]
    q = json.load(open(REC / "results/x3b/x3c_qwen3_8b_summary.json"))
    fr = {**{m: s[m]["frontier"] for m in ("llama31_8b", "mistral7b")}}
    qf = q.get("frontier") or q.get("per_model", {}).get("qwen3_8b", {}).get("frontier")
    fr["qwen3_8b"] = qf
    out = []
    for m, f in fr.items():
        for k, v in f.items():
            out.append({"model": LAB[m], "arm": k, "accuracy": v["accuracy"], "mean_calls": v["mean_calls"],
                        "mean_generated_tokens": v["mean_generated_tokens"]})
    wcsv("fig5_drspider_frontier.csv", out)


def fig6():
    c1 = list(csv.DictReader(open(REC / "tables/C1_comparisons.csv")))
    t1 = list(csv.DictReader(open(REC / "tables/T1_within_stack_decomposition.csv")))
    out = []
    for r in c1:
        if r["model"].startswith("Qwen3"):
            out.append({"model": r["model"], "set": r["set"], "comparison": r["comparison"], "n": r["n"], "chance": r["chance"],
                        "jit_acc": r["jit_acc"], "baseline_acc": r["baseline_acc"], "jit_tokens": r["jit_tokens_per_q"],
                        "baseline_tokens": r["baseline_tokens_per_q"], "diff": r["diff"], "lo": r["ci_lo"], "hi": r["ci_hi"]})
    wcsv("fig6_programs_vs_thinking.csv", out)
    out = []
    for r in t1:
        out.append({k: r[k] for k in ("model", "set", "n", "chance", "acc_direct", "acc_initial", "acc_generic", "acc_specific", "acc_cot", "acc_think")})
    wcsv("fig6_within_stack_decomposition.csv", out)


def fig3():
    """Fix-break ledger pooled over cells (from analysis/allocation_lens.json, written by allocation_lens.py)."""
    L = json.load(open(jp.outputs("analysis") / "allocation_lens.json"))
    P = L["pooled_N3"]; pw = L["proofwriter_revise_total"]
    rows = [("SQL", "self-consistency", P["uniform voting"], 3 * P["uniform voting"]["spent_on"], "samples"),
            ("SQL", "gated voting, winner replaces", P["gated voting"], 3 * P["gated voting"]["spent_on"], "samples"),
            ("SQL", "JIT-Repair, vote spend", P["JIT-Repair (vote)"], 3 * P["JIT-Repair (vote)"]["spent_on"], "samples"),
            ("SQL", "JIT-Repair, revise spend", P["JIT-Repair (revise)"], P["JIT-Repair (revise)"]["spent_on"], "revision calls"),
            ("ProofWriter", "JIT-Repair, revise spend", pw, "", "")]
    out = []
    for task, pol, r, cost, unit in rows:
        out.append({"task": task, "policy": pol, "n": r["n"], "fixes": r["fixes"], "breaks": r["breaks"],
                    "net": r["fixes"] - r["breaks"], "spent_on": r.get("spent_on", ""),
                    "spent_on_correct": r.get("spent_on_correct", ""), "extra_cost": cost, "cost_unit": unit})
    wcsv("fig3_ledger.csv", out)


def figS1():
    """Allocation curves on held-out BIRD (from analysis/allocation_lens_curves.csv), N=3."""
    rows = list(csv.DictReader(open(jp.outputs("analysis") / "allocation_lens_curves.csv")))
    th = {r["cell"]: r for r in csv.DictReader(open(jp.outputs("analysis") / "allocation_theory_checks.csv")) if r["N"] == "3"}
    keep = {"gate", "confidence (sum)", "random"}
    out = [{"model": LAB[r["cell"].split("/")[1]], "score": r["score"], "spend": r["spend"], "coverage": r["coverage"],
            "gain": r["gain"], "rho": th[r["cell"]]["rho"]}
           for r in rows if r["cell"].startswith("birddev_rest/") and r["score"] in keep]
    wcsv("figS1_allocation_curves.csv", out)


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5(); fig6(); figS1(); print(sorted(p.name for p in D.iterdir()))
