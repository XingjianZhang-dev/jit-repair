"""Numeric checks of the allocation statements (Supplement S2) on the stored per-question records.

For every within-stack SQL cell with stored vote samples (Spider test and BIRD Mini-Dev x Llama/Mistral; Dr.Spider
ablation subset x Llama/Mistral/Qwen3-8B; BIRD dev remainder x Llama/Mistral/Qwen3-8B/Qwen3-14B) and N in {2,3,5}:
  G_i   gate (frozen checker + critic flag of the greedy output)
  g_i   greedy correctness; w_i correctness of the exec-vote@N winner (= ungated vote@N outcome)
  a_i   frozen-selector acceptance of the winner against greedy
  D^U_i = w_i - g_i (winner always replaces greedy), D^S_i = a_i * D^U_i (winner passes the selector)
Statements checked (exact unless marked inequality):
  (T1) gated - random at the same rate = Cov(G, D) for D in {D^U, D^S}
  (T2) |Cov(G, D)| <= sqrt(rho(1-rho)) * sd(D) <= sqrt(rho(1-rho) * pi), pi = P(D != 0)          [inequality]
  (T3) gated - uniform = -E[(1-G) D^U]  (uniform = spend on every item, winner replaces greedy)
  (T4) selector-guarded gated - random(U) = Cov(G, D^S) + rho (E[D^S] - E[D^U])
  (T5) Prop 1 for the guarded spend: v(z*) >= v(z0); unflagged unchanged; accepted valid->valid lowers critic [counts]
  (T6) expected calls: gated = 1 + rho N; ungated = N + P(no executable sample among N)
Reads only; writes outputs/analysis/allocation_theory_checks.{csv,json}."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import csv, json, math, os, sys
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXP))
import x7_alloc as xa  # noqa: E402
from x7_confirm_load import load_model, MODELS, X7  # noqa: E402
OUT = jp.outputs("analysis")
REC = xa.REC


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def selections(name):
    b, m = name.split("/")
    if b in ("spider", "bird"):
        return json.load(open(REC / f"results/c2/{b}/{m}/selections_gold_free.json")), "c2"
    if b == "drspider_subset":
        return json.load(open(REC / f"results/x3b/{m}/selections_gold_free.json")), "x3b"
    return json.load(open(X7 / "birddev" / m / "selections_gold_free.json")), "x7"


def cells():
    out = list(xa.load_existing())
    for m in MODELS:
        out.append(load_model(m))
    return out


def cov(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(a * b) - np.mean(a) * np.mean(b))


def main():
    rows = []
    for name, ids, ini, win, pi, lp, qlen, clusters in cells():
        sel, kind = selections(name)
        for N in (2, 3, 5):
            G = np.array([1.0 if ini[c]["needs_repair"] else 0.0 for c in ids])
            g = np.array([float(pi[c]["greedy"]) for c in ids])
            w = np.array([float(pi[c][f"vote@{N}"]) for c in ids])
            a = np.array([1.0 if xa.accept(ini[c], *win[N][c]) else 0.0 for c in ids])
            DU = w - g; DS = a * DU
            rho = float(G.mean()); n = len(ids)
            ours, comps, f, k, gg = xa.vectors(ids, ini, win, pi, lp, qlen, N)
            verg = float(np.mean([ours[c] for c in ids]))
            veru = float(np.mean([comps["VER_U"][c] for c in ids]))
            randu = float(np.mean([comps["RAND_U"][c] for c in ids]))
            randg = float(np.mean([comps["RAND_G"][c] for c in ids]))
            uniform = float(w.mean())
            # T1
            t1u = (veru - randu) - cov(G, DU)
            t1s = (verg - randg) - cov(G, DS)
            # T2
            piU = float(np.mean(DU != 0)); piS = float(np.mean(DS != 0))
            b1u = math.sqrt(rho * (1 - rho)) * float(np.std(DU)); b2u = math.sqrt(rho * (1 - rho) * piU)
            b1s = math.sqrt(rho * (1 - rho)) * float(np.std(DS)); b2s = math.sqrt(rho * (1 - rho) * piS)
            # T1b: Cov(G, D) = rho (1 - rho) (E[D|G=1] - E[D|G=0]); corollary bound with outcome-change rates
            fl = G == 1
            def cond(D, m):
                return float(D[m].mean()) if m.any() else 0.0
            muU1, muU0, muS1, muS0 = cond(DU, fl), cond(DU, ~fl), cond(DS, fl), cond(DS, ~fl)
            t1bU = cov(G, DU) - rho * (1 - rho) * (muU1 - muU0)
            t1bS = cov(G, DS) - rho * (1 - rho) * (muS1 - muS0)
            piU1, piU0 = cond((DU != 0).astype(float), fl), cond((DU != 0).astype(float), ~fl)
            piS1, piS0 = cond((DS != 0).astype(float), fl), cond((DS != 0).astype(float), ~fl)
            corU = rho * (1 - rho) * (piU1 + piU0); corS = rho * (1 - rho) * (piS1 + piS0)
            # T3
            t3 = (veru - uniform) - (-float(np.mean((1 - G) * DU)))
            unfl = G == 0
            br = int(np.sum(unfl & (g == 1) & (w == 0))); fx = int(np.sum(unfl & (g == 0) & (w == 1)))
            brf = int(np.sum(~unfl & (g == 1) & (w == 0))); fxf = int(np.sum(~unfl & (g == 0) & (w == 1)))
            # T4
            t4 = (verg - randu) - (cov(G, DS) + rho * (float(DS.mean()) - float(DU.mean())))
            # T5 (guarded spend): validity of the selected output vs greedy, unflagged unchanged, critic on accepted
            v_viol = 0; unch_viol = 0; crit_viol = 0; vv_changes = 0
            for i, c in enumerate(ids):
                g0 = ini[c]; wsql, wvalid, wcrit = win[N][c]
                take = bool(G[i]) and xa.accept(g0, wsql, wvalid, wcrit)
                v_final = wvalid if take else bool(g0["execution_valid"])
                if v_final < bool(g0["execution_valid"]):
                    v_viol += 1
                if not G[i] and take:
                    unch_viol += 1
                if take and g0["execution_valid"] and wvalid:
                    vv_changes += 1
                    if not int(wcrit) < int(g0["critic_score"]):
                        crit_viol += 1
            # T6 calls
            gated_calls_formula = 1 + rho * N
            gated_calls_meas = float(np.mean([1 + (N if G[i] else 0) for i in range(n)]))
            ung_calls_meas = float(np.mean([sel[f"{c}|vote@{N}"][1 if kind == "x3b" else 2] for c in ids]))
            no_exec = float(np.mean([1.0 if sel[f"{c}|vote@{N}"][0] == ini[c]["sql"] and
                                     (sel[f"{c}|vote@{N}"][1 if kind == "x3b" else 2] == N + 1) else 0.0 for c in ids]))
            rows.append({"cell": name, "N": N, "n": n, "rho": rho, "greedy": float(g.mean()), "uniform_vote": uniform,
                         "gated_U": veru, "gated_S": verg, "random_U": randu, "random_S": randg,
                         "cov_G_DU": cov(G, DU), "cov_G_DS": cov(G, DS), "T1_resid_U": t1u, "T1_resid_S": t1s,
                         "pi_U": piU, "bound_sd_U": b1u, "bound_pi_U": b2u, "pi_S": piS, "bound_sd_S": b1s, "bound_pi_S": b2s,
                         "mu_U_flagged": muU1, "mu_U_passed": muU0, "mu_S_flagged": muS1, "mu_S_passed": muS0,
                         "T1b_resid_U": t1bU, "T1b_resid_S": t1bS,
                         "pi_U_flagged": piU1, "pi_U_passed": piU0, "pi_S_flagged": piS1, "pi_S_passed": piS0,
                         "corollary_bound_U": corU, "corollary_bound_S": corS,
                         "corollary_holds": bool(abs(cov(G, DU)) <= corU + 1e-12 and abs(cov(G, DS)) <= corS + 1e-12),
                         "T2_holds": bool(abs(cov(G, DU)) <= b1u + 1e-12 <= b2u + 2e-12 and abs(cov(G, DS)) <= b1s + 1e-12 <= b2s + 2e-12),
                         "E_DU_unflagged": float(DU[unfl].mean()) if unfl.any() else float("nan"),
                         "E_DU_flagged": float(DU[~unfl].mean()) if (~unfl).any() else float("nan"),
                         "unflagged_correct_broken": br, "unflagged_wrong_fixed": fx, "n_unflagged": int(unfl.sum()),
                         "unflagged_greedy_correct": int(np.sum(unfl & (g == 1))),
                         "flagged_correct_broken": brf, "flagged_wrong_fixed": fxf, "n_flagged": int((~unfl).sum()),
                         "flagged_fixed_selector": int(np.sum(fl & (DS > 0))), "flagged_broken_selector": int(np.sum(fl & (DS < 0))),
                         "flagged_fixed_noselector": int(np.sum(fl & (DU > 0))), "flagged_broken_noselector": int(np.sum(fl & (DU < 0))),
                         "gated_minus_uniform": veru - uniform, "minus_E_1mG_DU": -float(np.mean((1 - G) * DU)), "T3_resid": t3,
                         "T4_resid": t4, "selector_term": rho * (float(DS.mean()) - float(DU.mean())),
                         "T5_validity_violations": v_viol, "T5_unflagged_changed": unch_viol, "T5_valid_valid_changes": vv_changes,
                         "T5_critic_violations": crit_viol,
                         "calls_gated_formula": gated_calls_formula, "calls_gated_measured": gated_calls_meas,
                         "calls_ungated_measured": ung_calls_meas, "calls_ungated_formula": N + no_exec,
                         "calls_ratio": gated_calls_meas / ung_calls_meas})
    with open(OUT / "allocation_theory_checks.csv", "w", newline="") as h:
        wtr = csv.DictWriter(h, fieldnames=list(rows[0])); wtr.writeheader(); wtr.writerows(rows)
    worst = {k: max(abs(r[k]) for r in rows) for k in ("T1_resid_U", "T1_resid_S", "T1b_resid_U", "T1b_resid_S", "T3_resid", "T4_resid")}
    summ = {"cells": len(rows) // 3, "rows": len(rows), "max_abs_residuals": worst,
            "T2_holds_all": all(r["T2_holds"] for r in rows),
            "corollary_holds_all": all(r["corollary_holds"] for r in rows),
            "prop3_condition_holds_N3": sum(1 for r in rows if r["N"] == 3 and r["E_DU_unflagged"] <= 0),
            "T5_violations_total": sum(r["T5_validity_violations"] + r["T5_unflagged_changed"] + r["T5_critic_violations"] for r in rows),
            "N3_totals": {k: sum(r[k] for r in rows if r["N"] == 3) for k in ("flagged_fixed_selector", "flagged_broken_selector",
                          "flagged_fixed_noselector", "flagged_broken_noselector", "unflagged_wrong_fixed", "unflagged_correct_broken")},
            "T6_max_abs_gated_calls_err": max(abs(r["calls_gated_formula"] - r["calls_gated_measured"]) for r in rows),
            "T6_max_abs_ungated_calls_err": max(abs(r["calls_ungated_formula"] - r["calls_ungated_measured"]) for r in rows)}
    json.dump(summ, open(OUT / "allocation_theory_checks.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    for r in rows:
        if r["N"] == 3:
            print(f"{r['cell']:28s} rho={r['rho']:.3f} cov(G,DS)={100*r['cov_G_DS']:+.2f} corbound={100*r['corollary_bound_S']:.2f} pi1/pi0={r['pi_S_flagged']:.3f}/{r['pi_S_passed']:.3f} "
                  f"E[DU|G=0]={100*r['E_DU_unflagged']:+.2f} E[DU|G=1]={100*r['E_DU_flagged']:+.2f} broken/fixed(unfl)={r['unflagged_correct_broken']}/{r['unflagged_wrong_fixed']} "
                  f"of {r['unflagged_greedy_correct']} correct-unflagged; gated-uniform={100*r['gated_minus_uniform']:+.2f} calls {r['calls_gated_measured']:.2f}/{r['calls_ungated_measured']:.2f}={r['calls_ratio']:.2f}")


if __name__ == "__main__":
    main()
