"""X7 boundary table (descriptive): for the 7 discovery cells and the 4 confirmation cells, the allocation advantage of
VER-G at N=3 next to candidate moderators: trigger rate; flag precision / recall against greedy errors (gold,
descriptive); gold-free sample diversity (mean # distinct executable result keys among the first 3 samples; share of
items whose 3 samples are all executable and agree); and the spend comparison VER-G − JIT. Spearman rank correlations
over the 11 cells are reported as descriptive only (n = 11 cells, not independent: same models recur)."""
import csv, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x7_alloc as xa
from x7_confirm_load import load_model, MODELS
REC = xa.REC


def jl(p): return [json.loads(l) for l in open(p)]


def div(path):
    d3, ag = [], []
    for r in jl(path):
        k = [json.dumps(s["result_key"]) for s in r["samples"][:3] if s["execution_valid"] and s.get("key_status") == "ok"]
        d3.append(len(set(k))); ag.append(int(len(k) == 3 and len(set(k)) == 1))
    return float(np.mean(d3)), float(np.mean(ag))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


cells = []
for cell in xa.load_existing():
    name = cell[0]; b, m = name.split("/")
    p = REC / ("results/c2/%s/%s/samples_checked.jsonl" % (b, m) if b in ("spider", "bird") else "results/x3b/%s/samples_checked.jsonl" % m)
    cells.append(("discovery", cell, p))
for m in MODELS:
    cells.append(("confirmation", load_model(m), REC / f"results/x7/birddev/{m}/samples_checked.jsonl"))
rows = []
for stage, (name, ids, ini, win, pi, lp, qlen, clusters), sp in cells:
    ours, comps, f, k, g = xa.vectors(ids, ini, win, pi, lp, qlen, 3)
    cl = [clusters[c] for c in ids] if clusters else None
    flagged = [c for c in ids if ini[c]["needs_repair"]]; wrong = [c for c in ids if g[c] == 0]
    d3, ag = div(sp)
    r = {"stage": stage, "cell": name, "n": len(ids), "trigger_rate": f, "greedy_acc": float(np.mean(list(g.values()))),
         "flag_precision": float(np.mean([1 - g[c] for c in flagged])), "flag_recall": float(np.mean([ini[c]["needs_repair"] for c in wrong])),
         "distinct_keys_first3": d3, "all3_agree": ag}
    for key, other in (("RAND_U", comps["RAND_U"]), ("CONF_sum_U", comps["CONF_sum_U"]), ("JIT", {c: pi[c]["jit"] for c in ids})):
        mm, lo, hi = xa.boot([ours[c] - other[c] for c in ids], cl)
        r[f"VERG_minus_{key}"], r[f"VERG_minus_{key}_lo"], r[f"VERG_minus_{key}_hi"] = mm, lo, hi
    rows.append(r)
with open(REC / "tables/X7_boundary.csv", "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['stage'][:4]} {r['cell']:28s} f={r['trigger_rate']:.2f} prec={r['flag_precision']:.2f} rec={r['flag_recall']:.2f} d3={r['distinct_keys_first3']:.2f} agree={r['all3_agree']:.2f} "
          f"vsRAND={100*r['VERG_minus_RAND_U']:+.2f} vsCONF={100*r['VERG_minus_CONF_sum_U']:+.2f} vsJIT={100*r['VERG_minus_JIT']:+.2f} [{100*r['VERG_minus_JIT_lo']:+.2f},{100*r['VERG_minus_JIT_hi']:+.2f}]")
corr = {}
for y in ("VERG_minus_RAND_U", "VERG_minus_CONF_sum_U", "VERG_minus_JIT"):
    for x in ("trigger_rate", "flag_precision", "distinct_keys_first3", "all3_agree", "greedy_acc"):
        corr[f"{y}~{x}"] = spearman([r[x] for r in rows], [r[y] for r in rows])
json.dump({"n_cells": len(rows), "spearman": corr}, open(REC / "tables/X7_boundary_spearman.json", "w"), indent=1)
for k, v in corr.items(): print(f"{k:45s} rho={v:+.2f}")
