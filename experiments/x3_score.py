"""X3 scoring (gold): paper evaluators on combo final SQL and the vote winner; paired stats."""
from jit_repair import paths as jp  # noqa: E402
import argparse, json, multiprocessing as mp, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_analyze import _score, SAFE, INBOX, jl  # noqa
import paired_stats as st  # noqa
R = jp.OUTBOX

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); a = ap.parse_args()
    pool = mp.get_context("spawn").Pool(40); out = {}; pooled_c, pooled_v = {}, {}
    for f in sorted((R / "x3" / a.split).glob("*/*/combo_vote*.jsonl")):
        if f.name.endswith(".gen_repair.jsonl"):
            continue
        bench, model = f.parent.parent.name, f.parent.name
        safe = jl(SAFE[bench]); ids = [str(r["case_id"]) for r in safe]; pos = {c: i for i, c in enumerate(ids)}
        by = {str(r["case_id"]): r for r in safe}
        if bench == "spider":
            gold = {ids[i]: l.rstrip("\n").rsplit("\t", 1)[0] for i, l in enumerate(open(INBOX / "frozen/spider_llama31/prepared/gold.sql", encoding="utf-8"))}
        else:
            gold = {str(r["case_id"]): r["sql"] for r in jl(INBOX / "frozen/bird_assets/gold.jsonl")}
        recs = jl(f)
        vt = pool.map(_score, [(bench, pos[r["case_id"]], r["case_id"], by[r["case_id"]]["db_id"], r["vote_sql"], gold[r["case_id"]], bool(r["vote_valid"])) for r in recs], chunksize=8)
        ct = pool.map(_score, [(bench, pos[r["case_id"]], r["case_id"], by[r["case_id"]]["db_id"], r["final_sql"], gold[r["case_id"]], bool(r["final_valid"])) for r in recs], chunksize=8)
        v = {r["case_id"]: s for r, (s, _) in zip(recs, vt)}; c = {r["case_id"]: s for r, (s, _) in zip(recs, ct)}
        res = st.paired(c, v, seed=20261011)
        with open(f.with_suffix(".per_item.json"), "w") as hh:
            json.dump({"vote": v, "combo": c}, hh)
        pooled_c.update({f"{bench}|{model}|{k}": x for k, x in c.items()}); pooled_v.update({f"{bench}|{model}|{k}": x for k, x in v.items()})
        n = len(recs)
        res.update({"bench": bench, "model": model, "N": f.stem.replace("combo_vote", ""), "split": a.split,
                    "flagged": sum(r["flagged"] for r in recs),
                    "vote_mean_calls": sum(r["vote_calls"] for r in recs) / n, "combo_mean_calls": sum(r["combo_calls"] for r in recs) / n,
                    "vote_mean_tokens": sum(r["vote_tokens"] for r in recs) / n, "combo_mean_tokens": sum(r["combo_tokens"] for r in recs) / n})
        out[f"{bench}|{model}|N{res['N']}"] = res
        print(f"{bench:6s} {model:10s} N={res['N']} n={n} vote={res['baseline_accuracy']*100:.2f} combo={res['candidate_accuracy']*100:.2f} diff={res['difference']*100:+.2f} CI=[{res['ci95'][0]*100:+.2f},{res['ci95'][1]*100:+.2f}] p={res['mcnemar_exact_p']:.3g} rec/reg={res['candidate_only_correct']}/{res['baseline_only_correct']} flagged={res['flagged']} calls {res['vote_mean_calls']:.2f}->{res['combo_mean_calls']:.2f} tok {res['vote_mean_tokens']:.0f}->{res['combo_mean_tokens']:.0f}", flush=True)
    if a.split == "confirmation":
        pr = st.paired(pooled_c, pooled_v, seed=20261011)
        out["pooled_N3"] = pr
        out["a_all_cells_point_ge0"] = all(v_["difference"] >= 0 for k_, v_ in out.items() if k_.endswith("|N3"))
        out["b_pooled_ci_lo_gt0"] = pr["ci95"][0] > 0
        out["X3_verdict"] = "CONFIRMED" if (out["a_all_cells_point_ge0"] and out["b_pooled_ci_lo_gt0"]) else "NOT CONFIRMED"
        print("pooled", pr["n"], pr["difference"], pr["ci95"], out["X3_verdict"])
    json.dump(out, open(R / "x3" / f"x3_{a.split}_summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
