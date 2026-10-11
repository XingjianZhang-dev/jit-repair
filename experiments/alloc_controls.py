"""G2 (exploratory, not pre-registered): is the checker gate a better ALLOCATOR of sampling compute
than random or simple gold-free allocations at the same trigger rate / the same mean calls?
Offline, from existing per-item correctness (no new generation). Gold enters only through per-item correctness.

Policies per cell, N in {2,3,5} (vote@N over the same samples as C2/X3b):
  gated        treat = frozen checker+critic flag (needs_repair) -> vote@N, else greedy
  random@f     treat a uniformly random subset of the same size; evaluated in expectation per item
               (r_i = (1-f) g_i + f v_i), so no Monte-Carlo noise; same expected calls
  len_sql@f    treat the top-k items by greedy SQL output length (k = #flagged; ties by case_id hash)
  len_q@f      treat the top-k items by question length
  exec_only    treat only checker-INVALID greedy outputs (its own rate f_e), compared with random@f_e
  oracle@f     (gold-based upper bound, descriptive) treat greedy-wrong items first, up to k
  ungated@C    matched mean calls: per-item mixture of the two ungated frontier points (greedy, vote@1..8)
               bracketing the gated policy's mean calls
Stats: paired bootstrap 10,000, seed 20261011; Dr.Spider resamples the q_id clusters."""
from jit_repair import paths as jp  # noqa: E402
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
REC = jp.RECORDS
E = jp.ARCHIVE / "evidence"
def jl(p): return [json.loads(l) for l in open(p)]
SEED, B = 20261011, 10000


def boot(d, clusters=None):
    d = np.asarray(d, float); rng = np.random.default_rng(SEED)
    if clusters is None:
        idx = rng.integers(0, len(d), (B, len(d))); m = d[idx].mean(1)
    else:
        cl = sorted(set(clusters)); pos = {c: j for j, c in enumerate(cl)}
        s = np.zeros(len(cl)); n = np.zeros(len(cl))
        for x, c in zip(d, clusters): s[pos[c]] += x; n[pos[c]] += 1
        idx = rng.integers(0, len(cl), (B, len(cl))); m = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def h(c): return int(hashlib.sha256(c.encode()).hexdigest()[:8], 16)


def topk(ids, score, k):
    order = sorted(ids, key=lambda c: (-score[c], h(c)))
    return set(order[:k])


def cell(name, ids, g, v, calls_v, flag, invalid, sql_len, q_len, clusters):
    """g: greedy correct; v[N]: vote@N correct; calls_v[N]: per-item calls of ungated vote@N; flag/invalid: bool."""
    n = len(ids); k = sum(flag[c] for c in ids); f = k / n
    cl = [clusters[c] for c in ids] if clusters else None
    out = []
    for N in (2, 3, 5):
        gated = {c: (v[N][c] if flag[c] else g[c]) for c in ids}
        calls_g = np.mean([1 + N * flag[c] for c in ids])
        def comp(label, other, note=""):
            m, lo, hi = boot([gated[c] - other[c] for c in ids], cl)
            out.append({"cell": name, "N": N, "n": n, "trigger_rate": round(f, 4), "gated_acc": np.mean(list(gated.values())),
                        "comparator": label, "comparator_acc": float(np.mean([other[c] for c in ids])),
                        "gated_minus_comparator": m, "ci_lo": lo, "ci_hi": hi, "gated_mean_calls": float(calls_g), "note": note})
        comp("random@f (expectation)", {c: (1 - f) * g[c] + f * v[N][c] for c in ids}, "same expected calls")
        T = topk(ids, sql_len, k); comp("len_sql@f", {c: (v[N][c] if c in T else g[c]) for c in ids}, "top-k greedy SQL length")
        T = topk(ids, q_len, k); comp("len_question@f", {c: (v[N][c] if c in T else g[c]) for c in ids}, "top-k question length")
        wrong = {c: 1 - g[c] for c in ids}; T = topk(ids, wrong, k)
        comp("ORACLE@f (gold, upper bound)", {c: (v[N][c] if c in T else g[c]) for c in ids}, "descriptive upper bound")
        # exec-only gate at its own rate vs random at that rate
        ke = sum(invalid[c] for c in ids); fe = ke / n
        exec_only = {c: (v[N][c] if invalid[c] else g[c]) for c in ids}
        m, lo, hi = boot([exec_only[c] - ((1 - fe) * g[c] + fe * v[N][c]) for c in ids], cl)
        out.append({"cell": name, "N": N, "n": n, "trigger_rate": round(fe, 4), "gated_acc": float(np.mean(list(exec_only.values()))),
                    "comparator": "exec_only_gate vs random@f_exec", "comparator_acc": float(np.mean([(1 - fe) * g[c] + fe * v[N][c] for c in ids])),
                    "gated_minus_comparator": m, "ci_lo": lo, "ci_hi": hi, "gated_mean_calls": float(1 + N * fe), "note": "checker-only gate"})
        m, lo, hi = boot([gated[c] - exec_only[c] for c in ids], cl)
        out.append({"cell": name, "N": N, "n": n, "trigger_rate": round(f, 4), "gated_acc": float(np.mean(list(gated.values()))),
                    "comparator": "exec_only_gate (critic contribution)", "comparator_acc": float(np.mean(list(exec_only.values()))),
                    "gated_minus_comparator": m, "ci_lo": lo, "ci_hi": hi, "gated_mean_calls": float(calls_g), "note": "full gate minus checker-only gate (different rates)"})
        # matched-calls ungated mixture
        pts = [(1.0, g)] + [(float(np.mean([calls_v[M][c] for c in ids])), v[M]) for M in sorted(v)]
        pts.sort(key=lambda t: t[0])
        lo_pt = max([p for p in pts if p[0] <= calls_g], key=lambda t: t[0])
        hi_pt = min([p for p in pts if p[0] >= calls_g], key=lambda t: t[0])
        lam = 0.0 if hi_pt[0] == lo_pt[0] else (calls_g - lo_pt[0]) / (hi_pt[0] - lo_pt[0])
        mix = {c: (1 - lam) * lo_pt[1][c] + lam * hi_pt[1][c] for c in ids}
        comp("ungated vote mixture @ same mean calls", mix, f"mixture of points at {lo_pt[0]:.2f} and {hi_pt[0]:.2f} calls, lambda={lam:.2f}")
    return out


def main():
    rows = []
    # C2 cells (Spider/BIRD within-stack)
    SAFE = {"spider": E / "spider_generic_forward_canonical_v2_full/prepared/safe.jsonl",
            "bird": E / "bird_minidev_generic_forward_assets_v2/safe.jsonl"}
    for b in ("spider", "bird"):
        q = {str(r["case_id"]): r["question"] for r in jl(SAFE[b])}
        for m in ("llama31_8b", "mistral7b"):
            d = REC / "results/c2" / b / m
            ini = {r["case_id"]: r for r in jl(d / "initial.jsonl")}; pi = {r["case_id"]: r for r in jl(d / "per_item.jsonl")}
            sel = json.load(open(d / "selections_gold_free.json"))
            ids = list(ini)
            v = {N: {c: pi[c][f"vote@{N}"] for c in ids} for N in (1, 2, 3, 5, 8)}
            calls_v = {N: {c: sel[f"{c}|vote@{N}"][2] for c in ids} for N in (1, 2, 3, 5, 8)}
            rows += cell(f"{b}/{m}", ids, {c: pi[c]["greedy"] for c in ids}, v, calls_v,
                         {c: bool(ini[c]["needs_repair"]) for c in ids}, {c: not ini[c]["execution_valid"] for c in ids},
                         {c: ini[c]["output_tokens"] for c in ids}, {c: len(q[c]) for c in ids}, None)
    # Dr.Spider X3b/X3c cells
    safe = {r["case_id"]: r for r in jl(jp.RECORDS / "inputs/drspider/safe_subset.jsonl")}
    for m in ("llama31_8b", "mistral7b", "qwen3_8b"):
        d = REC / "results/x3b" / m
        if not (d / "per_item.jsonl").exists():
            print("missing per-item for", m); continue
        ini = {r["case_id"]: r for r in jl(d / "initial.jsonl")}; pi = {r["case_id"]: r for r in jl(d / "per_item.jsonl")}
        sel = json.load(open(d / "selections_gold_free.json")); ids = list(ini)
        v = {N: {c: pi[c][f"vote@{N}"] for c in ids} for N in (1, 2, 3, 5, 8)}
        calls_v = {N: {c: sel[f"{c}|vote@{N}"][1] for c in ids} for N in (1, 2, 3, 5, 8)}
        rows += cell(f"drspider_subset/{m}", ids, {c: pi[c]["greedy"] for c in ids}, v, calls_v,
                     {c: bool(ini[c]["needs_repair"]) for c in ids}, {c: not ini[c]["execution_valid"] for c in ids},
                     {c: ini[c]["output_tokens"] for c in ids}, {c: len(safe[c]["question"]) for c in ids},
                     {c: safe[c]["q_id_spider_dev"] for c in ids})
    with open(REC / "tables/G2_allocation_controls.csv", "w", newline="") as hdl:
        w = csv.DictWriter(hdl, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        if r["N"] == 3:
            print(f"{r['cell']:24s} f={r['trigger_rate']:.3f} gated={100*r['gated_acc']:.2f} vs {r['comparator'][:38]:38s} {100*r['comparator_acc']:.2f}  Δ={100*r['gated_minus_comparator']:+.2f} [{100*r['ci_lo']:+.2f},{100*r['ci_hi']:+.2f}]")


if __name__ == "__main__":
    main()
