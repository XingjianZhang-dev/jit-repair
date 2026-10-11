"""Per-item views of where test-time compute lands, from the stored per-question records (reads only).

For every within-stack SQL cell with stored vote samples and N in {2,3,5}:
  (L1) fix/break ledger: for each policy, the questions it fixes (first output wrong -> final right), breaks
       (right -> wrong) and the model calls it uses. Policies: uniform voting (winner replaces every first output),
       gated voting (winner replaces flagged outputs), gated voting behind the selector (JIT-Repair, vote spend),
       revision of flagged outputs (JIT-Repair, revise spend; N-independent).
  (L2) allocation curves: A_s(c) = (1/n) * sum of D over the ceil-free top c*n questions ranked by score s
       (ties broken uniformly at random, in expectation), for the gate and for every deployable comparator score.
       Random allocation is the chord c * mean(D); uniform spending is the endpoint A(1).
  (L3) gold-free change rates: delta_g = share of questions in gate group g whose spend output differs from the
       first output as a program (whitespace-normalised). Since D_i != 0 implies a changed program,
       pi_g <= delta_g, so |gated - random| <= rho(1-rho)(delta_1 + delta_0) can be computed without references.
  (L4) the same ledger for revision of failing logic programs on ProofWriter (C1 records, within-stack): a program
       that fails validation yields no answer, and a valid program is never replaced, so revision cannot break.
Writes analysis/allocation_lens_{ledger,curves,changes}.csv and allocation_lens.json."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import csv, json, re
from pathlib import Path
import numpy as np
import allocation_theory as at

OUT = jp.outputs("analysis")
xa = at.xa
GRID = [i / 100 for i in range(101)]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).rstrip(";").strip().lower()


def curve(D, score, ids):
    """Expected allocation curve: spend on the top fraction c by score (ties in expectation)."""
    n = len(ids)
    order = sorted(set(score.values()), reverse=True)
    blocks = [[i for i, c in enumerate(ids) if score[c] == v] for v in order]
    pts, cum, k = [(0.0, 0.0)], 0.0, 0
    for b in blocks:
        cum += float(np.sum(D[b])); k += len(b)
        pts.append((k / n, cum / n))
    xs, ys = zip(*pts)
    return [float(np.interp(c, xs, ys)) for c in GRID], pts


def main():
    ledger, curves, changes, summ = [], [], [], {}
    for name, ids, ini, win, pi, lp, qlen, clusters in at.cells():
        n = len(ids)
        G = np.array([1.0 if ini[c]["needs_repair"] else 0.0 for c in ids]); rho = float(G.mean())
        g = np.array([float(pi[c]["greedy"]) for c in ids])
        j = np.array([float(pi[c]["jit"]) for c in ids])
        for N in (2, 3, 5):
            w = np.array([float(pi[c][f"vote@{N}"]) for c in ids])
            a = np.array([1.0 if xa.accept(ini[c], *win[N][c]) else 0.0 for c in ids])
            changed = np.array([1.0 if norm(win[N][c][0]) != norm(ini[c]["sql"]) else 0.0 for c in ids])
            DU = w - g; DS = a * DU
            pols = {"uniform voting": (np.ones(n), DU), "gated voting": (G, G * DU),
                    "JIT-Repair (vote)": (G, G * DS), "JIT-Repair (revise)": (G, j - g)}
            for p, (S, D) in pols.items():
                if p == "JIT-Repair (revise)" and N != 3:
                    continue                      # revision does not depend on N; recorded once
                ledger.append({"cell": name, "N": N if p != "JIT-Repair (revise)" else "", "policy": p, "n": n,
                               "spent_on": int(S.sum()), "spent_on_correct": int((S * g).sum()),
                               "fixes": int(np.sum(D > 0)), "breaks": int(np.sum(D < 0))})
            # L3: gold-free change rates of the winner-replaces spend
            fl = G == 1
            d1 = float(changed[fl].mean()) if fl.any() else 0.0; d0 = float(changed[~fl].mean()) if (~fl).any() else 0.0
            p1 = float(np.mean(DU[fl] != 0)) if fl.any() else 0.0; p0 = float(np.mean(DU[~fl] != 0)) if (~fl).any() else 0.0
            covU = float(np.mean(G * DU) - G.mean() * DU.mean())
            # second gold-free bound: |gated - uniform| = (1 - rho)|mean D over passed| <= (1 - rho) delta_0
            dbar0 = float(DU[~fl].mean()) if (~fl).any() else 0.0
            changes.append({"cell": name, "N": N, "rho": rho, "delta_flagged": d1, "delta_passed": d0, "pi_flagged": p1,
                            "pi_passed": p0, "bound_gold_free": rho * (1 - rho) * (d1 + d0),
                            "bound_outcome": rho * (1 - rho) * (p1 + p0), "abs_cov": abs(covU),
                            "pi_le_delta": bool(p1 <= d1 + 1e-12 and p0 <= d0 + 1e-12),
                            "bound_holds": bool(abs(covU) <= rho * (1 - rho) * (d1 + d0) + 1e-12),
                            "uniform_gap": (1 - rho) * abs(dbar0), "uniform_bound": (1 - rho) * d0,
                            "uniform_bound_holds": bool((1 - rho) * abs(dbar0) <= (1 - rho) * d0 + 1e-12)})
            if N == 3:
                summ.setdefault("spend_choice_N3", {})[name] = {"jit_vote_acc": float(np.mean(g + G * DS)), "jit_revise_acc": float(np.mean(j)),
                                                                 "delta_flagged": d1, "vote_better": bool(np.mean(g + G * DS) > np.mean(j))}
                scores = {"gate": {c: float(G[i]) for i, c in enumerate(ids)},
                          "confidence (sum)": {c: -lp[c]["sum_lp"] for c in ids},
                          "confidence (mean)": {c: -lp[c]["mean_lp"] for c in ids},
                          "confidence (min)": {c: -lp[c]["min_lp"] for c in ids},
                          "length (question)": {c: float(qlen[c]) for c in ids},
                          "length (query)": {c: float(ini[c]["output_tokens"]) for c in ids}}
                for s, sc in scores.items():
                    ys, _ = curve(DU, sc, ids)
                    for cgrid, y in zip(GRID, ys):
                        curves.append({"cell": name, "score": s, "spend": "winner replaces", "coverage": cgrid, "gain": y})
                ys, _ = curve(DS, scores["gate"], ids)
                for cgrid, y in zip(GRID, ys):
                    curves.append({"cell": name, "score": "gate", "spend": "behind selector", "coverage": cgrid, "gain": y})
                for cgrid in GRID:
                    curves.append({"cell": name, "score": "random", "spend": "winner replaces", "coverage": cgrid, "gain": cgrid * float(DU.mean())})
    # pooled ledger
    pooled = {}
    for r in ledger:
        if r["N"] not in (3, ""):
            continue
        k = r["policy"]; p = pooled.setdefault(k, {"n": 0, "spent_on": 0, "spent_on_correct": 0, "fixes": 0, "breaks": 0})
        for f in p:
            p[f] += r[f]
    summ["pooled_N3"] = pooled
    U = pooled["uniform voting"]; V = pooled["JIT-Repair (vote)"]
    summ["fix_retention_vote_vs_uniform"] = V["fixes"] / U["fixes"]
    summ["break_share_vote_vs_uniform"] = V["breaks"] / U["breaks"]
    summ["sample_share_vote_vs_uniform"] = V["spent_on"] / U["spent_on"]
    summ["uniform_share_spent_on_correct"] = U["spent_on_correct"] / U["spent_on"]
    summ["gated_share_spent_on_correct"] = V["spent_on_correct"] / V["spent_on"]
    # L2: does the gate's curve lie above each comparator curve across coverage? (winner-replaces spend, N=3)
    dom = {}
    cv = {}
    for r in curves:
        cv.setdefault((r["cell"], r["score"], r["spend"]), []).append(r["gain"])
    cells_ = sorted({r["cell"] for r in curves})
    for s in ("confidence (sum)", "confidence (mean)", "confidence (min)", "length (question)", "length (query)", "random"):
        fr = []
        for c in cells_:
            gte = np.array(cv[(c, "gate", "winner replaces")]); oth = np.array(cv[(c, s, "winner replaces")])
            fr.append(float(np.mean(gte[1:-1] >= oth[1:-1] - 1e-12)))
        dom[s] = {"min_share_of_grid": min(fr), "cells_dominated_everywhere": int(sum(f == 1.0 for f in fr)), "per_cell": dict(zip(cells_, fr))}
    summ["gate_curve_dominance_N3"] = dom
    # L4: ProofWriter revision of failing programs (diagnostic revision vs first program), C1 records
    c1 = json.load(open(at.REC / "tables/C1_T1_analysis.json"))["t1"]
    pw = {k: {"n": v["specific_vs_initial"]["n"], "fixes": v["specific_vs_initial"]["candidate_only_correct"],
              "breaks": v["specific_vs_initial"]["baseline_only_correct"]} for k, v in c1.items()}
    summ["proofwriter_revise"] = pw
    summ["proofwriter_revise_total"] = {f: sum(v[f] for v in pw.values()) for f in ("n", "fixes", "breaks")}
    # change-rate summary at N=3 by model family (exploratory spend choice)
    ch3 = [r for r in changes if r["N"] == 3]
    fam = lambda r: "qwen3" if "qwen3" in r["cell"] else "llama_mistral"
    summ["delta_flagged_N3"] = {f: [min(r["delta_flagged"] for r in ch3 if fam(r) == f), max(r["delta_flagged"] for r in ch3 if fam(r) == f)]
                                for f in ("qwen3", "llama_mistral")}
    sc = summ["spend_choice_N3"]
    summ["spend_choice_separated_by_delta"] = {
        "vote_better_cells": sorted(c for c, v in sc.items() if v["vote_better"]),
        "revise_better_cells": sorted(c for c, v in sc.items() if not v["vote_better"]),
        "min_delta_where_vote_better": min(v["delta_flagged"] for v in sc.values() if v["vote_better"]),
        "max_delta_where_revise_better": max(v["delta_flagged"] for v in sc.values() if not v["vote_better"])}
    # scoring consistency: the X3 test re-scored the same three-sample vote winners as the C2 frontier (vote@3)
    dis = {}
    for b_, m in (("spider", "llama31_8b"), ("spider", "mistral7b"), ("bird", "llama31_8b"), ("bird", "mistral7b")):
        x3 = json.load(open(at.REC / f"results/x3/confirmation/{b_}/{m}/combo_vote3.per_item.json"))["vote"]
        c2 = {json.loads(l)["case_id"]: json.loads(l) for l in open(at.REC / f"results/c2/{b_}/{m}/per_item.jsonl") if l.strip()}
        dis[f"{b_}/{m}"] = sum(int(c2[c]["vote@3"]) != int(v) for c, v in x3.items())
    summ["x3_vs_c2_vote3_scoring_disagreements"] = dis
    summ["x3_vs_c2_vote3_scoring_disagreements_total"] = sum(dis.values())
    summ["uniform_bound_holds_all"] = all(r["uniform_bound_holds"] for r in changes)
    summ["gold_free_bound_holds_all"] = all(r["bound_holds"] for r in changes)
    summ["pi_le_delta_all"] = all(r["pi_le_delta"] for r in changes)
    summ["records"] = len(changes)
    for fn, rows in (("ledger", ledger), ("curves", curves), ("changes", changes)):
        with open(OUT / f"allocation_lens_{fn}.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    json.dump(summ, open(OUT / "allocation_lens.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
