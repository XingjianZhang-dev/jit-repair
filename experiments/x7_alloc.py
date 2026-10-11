"""X7: how good is the verifier gate as an ALLOCATOR of test-time compute, against random, length, the model's own
confidence and a gold oracle — under unguarded (U: winner replaces greedy) and guarded (G: frozen selector
choose(greedy, winner)) acceptance. Same trigger rate k/n for every allocation policy -> same calls (1 + N k/n).
Our method = VER-G (gate -> exec-vote@N -> frozen selector). CPU only; gold enters only via per-item correctness."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
REC = jp.RECORDS
from jit_repair.spider_critic_feedback import lint_question_sql, critic_score  # noqa
from jit_repair.bird_minidev_feedback import critic_text  # noqa
E = jp.ARCHIVE / "evidence"
SEED, B = 20261011, 10000
def jl(p): return [json.loads(l) for l in open(p)]
def h(c): return int(hashlib.sha256(c.encode()).hexdigest()[:8], 16)


def boot(d, clusters=None, level=0.95):
    d = np.asarray(d, float); rng = np.random.default_rng(SEED)
    if clusters is None:
        m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    else:
        cl = sorted(set(clusters)); pos = {c: j for j, c in enumerate(cl)}
        s = np.zeros(len(cl)); n = np.zeros(len(cl))
        for x, c in zip(d, clusters): s[pos[c]] += x; n[pos[c]] += 1
        idx = rng.integers(0, len(cl), (B, len(cl))); m = s[idx].sum(1) / n[idx].sum(1)
    a = round((1 - level) / 2, 10); lo, hi = np.quantile(m, [a, round(1 - a, 10)]); return float(d.mean()), float(lo), float(hi)


def accept(g, wsql, wvalid, wcrit):
    """frozen merge_spider_critic_retries.choose(previous=greedy, repair=winner) -> True if winner is taken"""
    if wsql == g["sql"]:
        return False
    if not g["execution_valid"]:
        return True
    if not wvalid:
        return False
    return int(wcrit) < int(g["critic_score"])


def topk(ids, score, k):  # highest score first, deterministic tie-break
    return set(sorted(ids, key=lambda c: (-score[c], h(c)))[:k])


def vectors(ids, ini, win, pi, lp, qlen, N):
    """Per-item correctness of VER-G (ours) and every comparator allocation at the same k, for exec-vote@N."""
    n = len(ids); k = sum(bool(ini[c]["needs_repair"]) for c in ids); f = k / n
    g = {c: pi[c]["greedy"] for c in ids}
    w = {c: pi[c][f"vote@{N}"] for c in ids}
    acc = {c: accept(ini[c], *win[N][c]) for c in ids}
    U = lambda T: {c: (w[c] if c in T else g[c]) for c in ids}
    G = lambda T: {c: (w[c] if (c in T and acc[c]) else g[c]) for c in ids}
    pols = {
        "VER": {c for c in ids if ini[c]["needs_repair"]},
        "CONF_mean": topk(ids, {c: -lp[c]["mean_lp"] for c in ids}, k),
        "CONF_min": topk(ids, {c: -lp[c]["min_lp"] for c in ids}, k),
        "CONF_sum": topk(ids, {c: -lp[c]["sum_lp"] for c in ids}, k),
        "LEN_sql": topk(ids, {c: ini[c]["output_tokens"] for c in ids}, k),
        "LEN_q": topk(ids, {c: qlen[c] for c in ids}, k),
        "ORACLE": topk(ids, {c: 1 - g[c] for c in ids}, k),
    }
    ours = G(pols["VER"])
    comps = {}
    for p, T in pols.items():
        for rule, fn in (("U", U), ("G", G)):
            if p == "VER" and rule == "G":
                continue
            comps[f"{p}_{rule}"] = fn(T)
    # random allocation, in expectation over subsets of size k
    comps["RAND_U"] = {c: (1 - f) * g[c] + f * w[c] for c in ids}
    comps["RAND_G"] = {c: (1 - f) * g[c] + f * (w[c] if acc[c] else g[c]) for c in ids}
    return ours, comps, f, k, g


def analyze_cell(name, ids, ini, win, pi, lp, qlen, clusters, Ns=(2, 3, 5)):
    n = len(ids); cl = [clusters[c] for c in ids] if clusters else None
    rows = []
    for N in Ns:
        ours, comps, f, k, g = vectors(ids, ini, win, pi, lp, qlen, N)
        res = {"cell": name, "N": N, "n": n, "trigger_rate": round(f, 4), "greedy": float(np.mean(list(g.values()))),
               "VER_G": float(np.mean(list(ours.values()))), "calls": 1 + N * f}
        for key, other in comps.items():
            m, lo, hi = boot([ours[c] - other[c] for c in ids], cl)
            rows.append({**res, "comparator": key, "comparator_acc": float(np.mean(list(other.values()))),
                         "VER_G_minus": m, "ci_lo": lo, "ci_hi": hi})
    return rows


def load_existing():
    cells = []
    SAFE = {"spider": E / "spider_generic_forward_canonical_v2_full/prepared/safe.jsonl", "bird": E / "bird_minidev_generic_forward_assets_v2/safe.jsonl"}
    for b in ("spider", "bird"):
        rows = {str(r["case_id"]): r for r in jl(SAFE[b])}
        for m in ("llama31_8b", "mistral7b"):
            d = REC / "results/c2" / b / m
            ini = {r["case_id"]: r for r in jl(d / "initial.jsonl")}; pi = {r["case_id"]: r for r in jl(d / "per_item.jsonl")}
            sel = json.load(open(d / "selections_gold_free.json")); ids = list(ini)
            def crit(c, s):
                q = critic_text(str(rows[c]["question"]), str(rows[c]["evidence"])) if b == "bird" else str(rows[c]["question"])
                return critic_score(lint_question_sql(q, s))
            win = {N: {c: (sel[f"{c}|vote@{N}"][0], bool(sel[f"{c}|vote@{N}"][1]), crit(c, sel[f"{c}|vote@{N}"][0])) for c in ids} for N in (2, 3, 5)}
            lp = {r["case_id"]: r for r in jl(REC / f"results/x7/logprobs/{b}__{m}.jsonl")}
            cells.append((f"{b}/{m}", ids, ini, win, pi, lp, {c: len(rows[c]["question"]) for c in ids}, None))
    safe = {r["case_id"]: r for r in jl(jp.RECORDS / "inputs/drspider/safe_subset.jsonl")}
    for m in ("llama31_8b", "mistral7b", "qwen3_8b"):
        d = REC / "results/x3b" / m
        ini = {r["case_id"]: r for r in jl(d / "initial.jsonl")}; pi = {r["case_id"]: r for r in jl(d / "per_item.jsonl")}
        sel = json.load(open(d / "selections_gold_free.json")); ids = list(ini)
        def wv(c, N):
            s = sel[f"{c}|vote@{N}"][0]
            v = ini[c]["execution_valid"] if s == ini[c]["sql"] else True
            return s, v, critic_score(lint_question_sql(str(safe[c]["question"]), s))
        win = {N: {c: wv(c, N) for c in ids} for N in (2, 3, 5)}
        lpp = REC / f"results/x7/logprobs/drspider__{m}.jsonl"
        if not lpp.exists():
            print("no logprobs", m); continue
        lp = {r["case_id"]: r for r in jl(lpp)}
        cells.append((f"drspider_subset/{m}", ids, ini, win, pi, lp, {c: len(safe[c]["question"]) for c in ids},
                      {c: safe[c]["q_id_spider_dev"] for c in ids}))
    return cells


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    rows = []
    if which == "discovery":
        for cell in load_existing():
            rows += analyze_cell(*cell)
        out = REC / "tables/X7_discovery_allocation.csv"
    else:
        from x7_confirm_load import load_birddev  # noqa
        for cell in load_birddev():
            rows += analyze_cell(*cell)
        out = REC / "tables/X7_confirmation_by_cell.csv"  # discovery-format table; X7_confirmation_allocation.csv comes from x7_confirm.py
    with open(out, "w", newline="") as hdl:
        wtr = csv.DictWriter(hdl, fieldnames=list(rows[0])); wtr.writeheader(); wtr.writerows(rows)
    for r in rows:
        if r["N"] == 3 and r["comparator"] in ("RAND_U", "RAND_G", "CONF_mean_U", "CONF_mean_G", "CONF_min_U", "CONF_min_G", "CONF_sum_U", "VER_U", "ORACLE_G", "ORACLE_U"):
            print(f"{r['cell']:24s} f={r['trigger_rate']:.3f} VER-G={100*r['VER_G']:.2f} vs {r['comparator']:12s} {100*r['comparator_acc']:.2f} Δ={100*r['VER_G_minus']:+.2f} [{100*r['ci_lo']:+.2f},{100*r['ci_hi']:+.2f}]")


if __name__ == "__main__":
    main()
