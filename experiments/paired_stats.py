"""Paired statistics: bootstrap (item or cluster resampling), exact McNemar, Holm.
Bootstrap mirrors the frozen jit_repair.paired_statistics (numpy default_rng(seed).integers over
sorted ids, np.quantile [0.025, 0.975])."""
from __future__ import annotations

import math
import numpy as np


def exact_mcnemar_p(wins: int, losses: int) -> float:
    d = wins + losses
    if d == 0:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(d, i) for i in range(tail + 1)) / (1 << d))


def paired(cand: dict, base: dict, seed: int = 20261011, samples: int = 10_000, clusters: dict | None = None):
    """cand/base: {id: 0/1 correct}. clusters: optional {id: cluster_id} for cluster bootstrap."""
    if set(cand) != set(base):
        raise ValueError("paired arms have different ids")
    ids = sorted(cand)
    a = np.array([int(base[i]) for i in ids], dtype=float)
    b = np.array([int(cand[i]) for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(samples)
    if clusters is None:
        n = len(ids)
        for k in range(samples):
            idx = rng.integers(0, n, n)
            diffs[k] = b[idx].mean() - a[idx].mean()
    else:
        cl = sorted({clusters[i] for i in ids})
        pos = {c: j for j, c in enumerate(cl)}
        sa = np.zeros(len(cl)); sb = np.zeros(len(cl)); cnt = np.zeros(len(cl))
        for i, x, y in zip(ids, a, b):
            j = pos[clusters[i]]; sa[j] += x; sb[j] += y; cnt[j] += 1
        for k in range(samples):
            idx = rng.integers(0, len(cl), len(cl))
            diffs[k] = sb[idx].sum() / cnt[idx].sum() - sa[idx].sum() / cnt[idx].sum()
    wins = int(np.sum((b == 1) & (a == 0)))
    losses = int(np.sum((b == 0) & (a == 1)))
    lo, hi = np.quantile(diffs, [0.025, 0.975]).tolist()
    return {"n": len(ids), "baseline_accuracy": float(a.mean()), "candidate_accuracy": float(b.mean()),
            "difference": float(b.mean() - a.mean()), "ci95": [lo, hi], "candidate_only_correct": wins,
            "baseline_only_correct": losses, "mcnemar_exact_p": exact_mcnemar_p(wins, losses),
            "bootstrap_samples": samples, "seed": seed, "cluster_bootstrap": clusters is not None}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items); out = {}; running = 0.0
    for r, (k, p) in enumerate(items):
        adj = min(1.0, (m - r) * p)
        running = max(running, adj)
        out[k] = running
    return out
