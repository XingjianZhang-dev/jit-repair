"""Selector-matched allocation comparison on held-out BIRD (descriptive; reads the X7 records only).

JIT-Repair (gate, vote of N, selector) minus each allocation policy at the same number of flagged questions, with the
policy's vote winner either replacing the output (no selector) or passing the same selector. Per model and pooled over
the four models (paired bootstrap, 10,000 resamples; pooled intervals resample question ids so that a question's
outputs from all models stay together). Uses the loader and bootstrap of experiments/x7_confirm.py unchanged.
N = 2, 3, 5. Writes analysis/selector_matched.csv."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import csv, os, sys
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXP))
import x7_alloc as xa  # noqa: E402
from x7_confirm_load import load_model, MODELS  # noqa: E402
OUT = jp.outputs("analysis")

COMPS = ["RAND_U", "RAND_G", "CONF_sum_U", "CONF_sum_G", "LEN_q_U", "LEN_q_G", "ORACLE_U", "ORACLE_G", "VER_U"]

def main() -> None:
    cells = {m: load_model(m) for m in MODELS}
    rows = []
    for N in (2, 3, 5):
        rows += one_n(cells, N)
    with open(OUT / "selector_matched.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def one_n(cells, N):
    per = {}
    for m in MODELS:
        name, ids, ini, win, pi, lp, qlen, _ = cells[m]
        ours, comps, f, k, g = xa.vectors(ids, ini, win, pi, lp, qlen, N)
        per[m] = (ids, ours, comps, f)
    rows = []
    for c in COMPS:
        for m in MODELS:
            ids, ours, comps, f = per[m]
            mm, lo, hi = xa.boot([ours[i] - comps[c][i] for i in ids])
            rows.append({"N": N, "model": m, "comparator": c, "rho": f, "acc_jit": float(np.mean([ours[i] for i in ids])),
                         "acc_comp": float(np.mean([comps[c][i] for i in ids])), "diff": mm, "ci_lo": lo, "ci_hi": hi})
        d, cl = [], []
        for m in MODELS:
            ids, ours, comps, f = per[m]
            d += [ours[i] - comps[c][i] for i in ids]; cl += ids
        mm, lo, hi = xa.boot(d, cl, 0.95)
        rows.append({"N": N, "model": "POOLED", "comparator": c, "rho": float("nan"), "acc_jit": float("nan"),
                     "acc_comp": float("nan"), "diff": mm, "ci_lo": lo, "ci_hi": hi})
    return rows

if __name__ == "__main__":
    main()
