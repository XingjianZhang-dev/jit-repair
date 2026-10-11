"""G3 (exploratory): does trigger-gated voting keep Proposition 1 (pointwise checker safety)?
Proposition 1 of the paper: (i) v(z*) >= v(z0) for every input; (ii) unflagged outputs are unchanged; (iii) if v0 = v1 = 1 and
the replacement is accepted, then c(z*) < c(z0) (critic score strictly decreases).
Checked empirically for: JIT (within-stack regeneration), gated-vote@N, and a GUARDED gated vote that passes the
vote winner through the frozen selector `choose(greedy, winner)`. Critic scores of winners are recomputed gold-free
with the frozen lint (Spider/Dr.Spider: question; BIRD: critic_text(question, evidence))."""
from jit_repair import paths as jp  # noqa: E402
import csv, json, os, sys
from pathlib import Path
import numpy as np
REC = jp.RECORDS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jit_repair.spider_critic_feedback import lint_question_sql, critic_score  # noqa
from jit_repair.bird_minidev_feedback import critic_text  # noqa
from jit_repair.merge_spider_critic_retries import choose  # noqa
import paired_stats as st  # noqa
E = jp.ARCHIVE / "evidence"
def jl(p): return [json.loads(l) for l in open(p)]


def crit(bench, row, sql):
    q = critic_text(str(row["question"]), str(row["evidence"])) if bench == "bird" else str(row["question"])
    return critic_score(lint_question_sql(q, sql))


def run_cell(name, bench, ids, rows, ini, fin, pi, winner, clusters):
    """winner[N][c] = (sql, valid) of the ungated vote@N winner; pi: per-item correctness incl. greedy, jit, vote@N."""
    out = []
    # JIT
    viol_i = sum(ini[c]["execution_valid"] and not fin[c]["execution_valid"] for c in ids)
    viol_ii = sum((not ini[c]["needs_repair"]) and fin[c]["sql"] != ini[c]["sql"] for c in ids)
    vv = [c for c in ids if ini[c]["execution_valid"] and fin[c]["execution_valid"] and fin[c]["sql"] != ini[c]["sql"]]
    viol_iii = sum(int(fin[c]["critic_score"]) >= int(ini[c]["critic_score"]) for c in vv)
    out.append({"cell": name, "method": "JIT (within-stack)", "n": len(ids), "flagged": sum(ini[c]["needs_repair"] for c in ids),
                "viol_i_valid_to_invalid": viol_i, "viol_ii_unflagged_changed": viol_ii, "valid_to_valid_changes": len(vv),
                "viol_iii_critic_not_decreased": viol_iii})
    for N in (2, 3, 5):
        gated_sql, gated_valid, guarded, gcorr, gdcorr = {}, {}, {}, {}, {}
        n_vv = n_iii = n_i = n_ii = 0; guard_rej = 0
        for c in ids:
            g = ini[c]
            if not g["needs_repair"]:
                s, v = g["sql"], g["execution_valid"]; gcorr[c] = pi[c]["greedy"]; gdcorr[c] = pi[c]["greedy"]
                gated_sql[c], gated_valid[c] = s, v; continue
            s, v = winner[N][c]
            gated_sql[c], gated_valid[c] = s, v; gcorr[c] = pi[c][f"vote@{N}"]
            n_i += int(g["execution_valid"] and not v)
            if g["execution_valid"] and v and s != g["sql"]:
                n_vv += 1; cw = crit(bench, rows[c], s); n_iii += int(cw >= int(g["critic_score"]))
            # guarded: frozen selector between greedy and winner
            cand = {"execution_valid": v, "critic_score": crit(bench, rows[c], s) if v else 0, "sql": s}
            sel, reason = choose(g, cand)
            if sel is g or sel["sql"] == g["sql"]:
                gdcorr[c] = pi[c]["greedy"]; guard_rej += int(s != g["sql"])
            else:
                gdcorr[c] = pi[c][f"vote@{N}"]
        cl = None if clusters is None else clusters
        a = st.paired(gdcorr, gcorr, seed=20261011, clusters=cl)
        b = st.paired(gdcorr, {c: pi[c]["jit"] for c in ids}, seed=20261011, clusters=cl)
        out.append({"cell": name, "method": f"gated_vote@{N}", "n": len(ids), "flagged": sum(ini[c]["needs_repair"] for c in ids),
                    "viol_i_valid_to_invalid": n_i, "viol_ii_unflagged_changed": 0, "valid_to_valid_changes": n_vv,
                    "viol_iii_critic_not_decreased": n_iii,
                    "gated_acc": float(np.mean(list(gcorr.values()))), "guarded_acc": float(np.mean(list(gdcorr.values()))),
                    "guard_rejected_winners": guard_rej,
                    "guarded_minus_gated": a["difference"], "gg_ci_lo": a["ci95"][0], "gg_ci_hi": a["ci95"][1],
                    "guarded_minus_jit": b["difference"], "gj_ci_lo": b["ci95"][0], "gj_ci_hi": b["ci95"][1]})
    return out


def main():
    rows_out = []
    SAFE = {"spider": E / "spider_generic_forward_canonical_v2_full/prepared/safe.jsonl", "bird": E / "bird_minidev_generic_forward_assets_v2/safe.jsonl"}
    for b in ("spider", "bird"):
        rows = {str(r["case_id"]): r for r in jl(SAFE[b])}
        for m in ("llama31_8b", "mistral7b"):
            d = REC / "results/c2" / b / m
            ini = {r["case_id"]: r for r in jl(d / "initial.jsonl")}; fin = {r["case_id"]: r for r in jl(d / "final.jsonl")}
            pi = {r["case_id"]: r for r in jl(d / "per_item.jsonl")}; sel = json.load(open(d / "selections_gold_free.json"))
            ids = list(ini)
            winner = {N: {c: (sel[f"{c}|vote@{N}"][0], bool(sel[f"{c}|vote@{N}"][1])) for c in ids} for N in (2, 3, 5)}
            rows_out += run_cell(f"{b}/{m}", b, ids, rows, ini, fin, pi, winner, None)
    rows = {r["case_id"]: r for r in jl(jp.RECORDS / "inputs/drspider/safe_subset.jsonl")}
    for m in ("llama31_8b", "mistral7b", "qwen3_8b"):
        d = REC / "results/x3b" / m
        if not (d / "per_item.jsonl").exists():
            print("missing", m); continue
        ini = {r["case_id"]: r for r in jl(d / "initial.jsonl")}; fin = {r["case_id"]: r for r in jl(d / "final.jsonl")}
        pi = {r["case_id"]: r for r in jl(d / "per_item.jsonl")}; sel = json.load(open(d / "selections_gold_free.json"))
        ids = list(ini)
        def w(c, N):
            s = sel[f"{c}|vote@{N}"][0]
            return s, (ini[c]["execution_valid"] if s == ini[c]["sql"] else True)
        winner = {N: {c: w(c, N) for c in ids} for N in (2, 3, 5)}
        rows_out += run_cell(f"drspider_subset/{m}", "drspider", ids, rows, ini, fin, pi, winner,
                             {c: rows[c]["q_id_spider_dev"] for c in ids})
    keys = sorted({k for r in rows_out for k in r}, key=lambda k: list(rows_out[1].keys()).index(k) if k in rows_out[1] else 99)
    with open(REC / "tables/G3_prop1_checks.csv", "w", newline="") as h:
        wtr = csv.DictWriter(h, fieldnames=keys); wtr.writeheader(); wtr.writerows(rows_out)
    for r in rows_out:
        extra = ""
        if "guarded_acc" in r:
            extra = (f" gated={100*r['gated_acc']:.2f} guarded={100*r['guarded_acc']:.2f} (rej {r['guard_rejected_winners']}) "
                     f"guarded-gated={100*r['guarded_minus_gated']:+.2f}[{100*r['gg_ci_lo']:+.2f},{100*r['gg_ci_hi']:+.2f}] "
                     f"guarded-JIT={100*r['guarded_minus_jit']:+.2f}[{100*r['gj_ci_lo']:+.2f},{100*r['gj_ci_hi']:+.2f}]")
        print(f"{r['cell']:24s} {r['method']:18s} (i)={r['viol_i_valid_to_invalid']} (ii)={r['viol_ii_unflagged_changed']} vv={r['valid_to_valid_changes']} (iii)viol={r['viol_iii_critic_not_decreased']}{extra}")


if __name__ == "__main__":
    main()
