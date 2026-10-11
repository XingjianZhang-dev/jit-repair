"""X7 confirmation loader: BIRD dev remainder (1,033 never-used questions) cells, in the same tuple format as
x7_alloc.load_existing(): (name, ids, ini, win, pi, lp, qlen, clusters).
win[N][case] = (winner_sql, winner_checker_valid, winner_critic_score) read from selections_gold_free.json
(elements [0], [1], [4] -- all gold-free, fixed before scoring)."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import json, os
from pathlib import Path
REC = jp.RECORDS
X7 = REC / "results/x7"
MODELS = ("llama31_8b", "mistral7b", "qwen3_8b", "qwen3_14b")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_model(m, Ns=(2, 3, 5)):
    rows = jl(X7 / "birddev_assets/safe.jsonl"); safe = {r["case_id"]: r for r in rows}; ids = [r["case_id"] for r in rows]
    d = X7 / "birddev" / m
    ini = {r["case_id"]: r for r in jl(d / "initial.jsonl")}
    pi = {r["case_id"]: r for r in jl(d / "per_item.jsonl")}
    sel = json.load(open(d / "selections_gold_free.json"))
    assert set(ids) == set(ini) == set(pi), m
    win = {N: {c: (sel[f"{c}|vote@{N}"][0], bool(sel[f"{c}|vote@{N}"][1]), int(sel[f"{c}|vote@{N}"][4])) for c in ids} for N in Ns}
    lp = {r["case_id"]: r for r in jl(X7 / f"logprobs/birddev__{m}.jsonl")}
    assert set(lp) == set(ids), m
    return (f"birddev_rest/{m}", ids, ini, win, pi, lp, {c: len(safe[c]["question"]) for c in ids}, None)


def load_birddev(models=MODELS):
    return [load_model(m) for m in models]
