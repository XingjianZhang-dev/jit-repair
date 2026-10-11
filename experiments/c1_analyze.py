"""C1 and the within-stack accuracy decomposition (Table S11) on the ProofWriter runs (CPU). Recomputes every label from raw
outputs with the pre-registered extractor (pw_tools.extract_label); strict parser as sensitivity."""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pw_tools as pt  # noqa
import paired_stats as st  # noqa

LAB4 = {"true", "false", "both", "unknown"}
CHANCE = {"conf": 0.25, "d5": 1 / 3}
FAMILY = [("qwen3_14b", "conf", "think"), ("qwen3_14b", "conf", "cot"), ("qwen3_14b", "d5", "think"),
          ("qwen3_14b", "d5", "cot"), ("qwen3_8b", "d5", "think"), ("qwen3_8b", "d5", "cot"),
          ("llama31_8b", "d5", "cot"), ("mistral7b", "d5", "cot")]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def corr(rows, key="prediction"):
    return {r["case_id"]: int(r[key] in LAB4 and r[key] == r["gold"]) for r in rows}


def arm_reasoning(d, mode):
    p = d / f"{mode}.jsonl"
    if not p.exists():
        return None
    rows = jl(p)
    for r in rows:  # recompute from raw with the pre-registered extractor
        r["prediction_recomputed"] = pt.extract_label(r["raw_output"])
        r["strict_recomputed"] = pt.strict_label(r["raw_output"])
    trunc = json.load(open(d / f"{mode}.truncation.json"))
    n = len(rows)
    t_final = sum(r["finish_reason"] == "length" for r in rows)
    return {"rows": rows, "correct": corr(rows, "prediction_recomputed"), "strict_correct": corr(rows, "strict_recomputed"),
            "truncated_final": t_final, "truncation_rate": t_final / n, "truncation_info": trunc,
            "no_answer": sum(r["prediction_recomputed"] is None for r in rows),
            "mean_tokens": sum(r["output_tokens"] for r in rows) / n,
            "extractor_matches_generation_time": sum(r["prediction_recomputed"] == r["prediction"] for r in rows)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root); out = {"comparisons": {}, "arms": {}, "t1": {}}
    pvals = {}
    for model, s, base in FAMILY:
        d = root / s / model
        key = f"{model}|{s}|JIT-{base}"
        jitp = d / "dsl_specific_merged_3.jsonl"
        b = arm_reasoning(d, base)
        if not jitp.exists() or b is None:
            out["comparisons"][key] = {"status": "MISSING"}; continue
        jit = jl(jitp); jc = corr(jit)
        evaluable = b["truncation_rate"] < 0.02
        res = st.paired(jc, b["correct"], seed=20261011)
        res.update({"set": s, "chance": CHANCE[s], "baseline": base, "baseline_truncation_rate": b["truncation_rate"],
                    "evaluable": evaluable, "baseline_no_answer": b["no_answer"],
                    "jit_mean_total_output_tokens": sum(r.get("total_output_tokens", r["output_tokens"]) for r in jit) / len(jit),
                    "baseline_mean_output_tokens": b["mean_tokens"],
                    "strict_parser_sensitivity": st.paired(jc, b["strict_correct"], seed=20261011)})
        out["comparisons"][key] = res
        if evaluable:
            pvals[key] = res["mcnemar_exact_p"]
    holm = st.holm(pvals)
    verdicts = []
    for k, r in out["comparisons"].items():
        if r.get("status") == "MISSING":
            r["verdict"] = "NOT EVALUABLE (missing)"; verdicts.append("NE"); continue
        if not r["evaluable"]:
            r["verdict"] = "NOT EVALUABLE (truncation >= 2% after declared increase)"; verdicts.append("NE"); continue
        r["holm_p"] = holm[k]
        ok = r["difference"] > 0 and r["ci95"][0] > 0 and r["holm_p"] < 0.05
        r["verdict"] = "PASS" if ok else "FAIL"; verdicts.append(r["verdict"])
    out["C1_overall"] = ("CONFIRMED" if all(v == "PASS" for v in verdicts) else
                         "NOT CONFIRMED" if "FAIL" in verdicts else "PARTIALLY EVALUABLE")
    # within-stack T1 decomposition + arms summary
    for s in ("conf", "d5"):
        for model in ("qwen3_14b", "qwen3_8b", "llama31_8b", "mistral7b"):
            d = root / s / model
            if not (d / "dsl_initial.jsonl").exists():
                continue
            arms = {}
            for nm, f in [("direct", "direct.jsonl"), ("initial", "dsl_initial.jsonl"),
                          ("generic_r1", "dsl_generic_merged_1.jsonl"), ("generic", "dsl_generic_merged_3.jsonl"),
                          ("specific_r1", "dsl_specific_merged_1.jsonl"), ("specific", "dsl_specific_merged_3.jsonl")]:
                if (d / f).exists():
                    arms[nm] = corr(jl(d / f))
            for nm in ("cot", "think"):
                r = arm_reasoning(d, nm)
                if r:
                    arms[nm] = r["correct"]
                    out["arms"][f"{model}|{s}|{nm}"] = {k: v for k, v in r.items() if k not in ("rows", "correct", "strict_correct")}
                    out["arms"][f"{model}|{s}|{nm}"]["accuracy"] = sum(r["correct"].values()) / len(r["correct"])
                    out["arms"][f"{model}|{s}|{nm}"]["strict_accuracy"] = sum(r["strict_correct"].values()) / len(r["strict_correct"])
            n = len(arms["initial"])
            t1 = {"n": n, "chance": CHANCE[s], "acc": {k: sum(v.values()) / n for k, v in arms.items()}}
            pairs = [("formalisation", "initial", "direct"), ("retry", "generic", "initial"),
                     ("diagnostic", "specific", "generic"), ("retry_brief", "generic", "direct"),
                     ("diagnostic_round1", "specific_r1", "generic_r1"), ("specific_vs_initial", "specific", "initial")]
            for nm, c, b in pairs:
                if c in arms and b in arms:
                    t1[nm] = st.paired(arms[c], arms[b], seed=20261011)
            out["t1"][f"{model}|{s}"] = t1
    json.dump(out, open(a.out, "w"), indent=1)
    for k, r in out["comparisons"].items():
        if "difference" in r:
            print(f"{k:32s} n={r['n']} JIT={r['candidate_accuracy']*100:.1f} base={r['baseline_accuracy']*100:.1f} "
                  f"diff={r['difference']*100:+.1f} CI=[{r['ci95'][0]*100:.1f},{r['ci95'][1]*100:.1f}] holm={r.get('holm_p')} "
                  f"trunc={r['baseline_truncation_rate']*100:.1f}% tok JIT/base={r['jit_mean_total_output_tokens']:.0f}/{r['baseline_mean_output_tokens']:.0f} -> {r['verdict']}")
        else:
            print(k, r)
    print("C1_overall:", out["C1_overall"])
    for k, t in out["t1"].items():
        print(k, {a_: round(v * 100, 1) for a_, v in t["acc"].items()})


if __name__ == "__main__":
    main()
