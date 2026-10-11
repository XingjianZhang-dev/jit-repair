"""X2 confirmation: diagnostic value by initial error type on the within-stack depth-5 runs (4 models)."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_stats as st  # noqa
from pw_errors import classify  # noqa
root = Path(sys.argv[1]); out_path = Path(sys.argv[2])
def jl(p): return [json.loads(l) for l in open(p)]
res = {}; pv = {}
for m in ("qwen3_14b", "qwen3_8b", "llama31_8b", "mistral7b"):
    d = root / "d5" / m
    ini = {r["case_id"]: r for r in jl(d / "dsl_initial.jsonl")}
    sp = {r["case_id"]: int(r["prediction"] == r["gold"]) for r in jl(d / "dsl_specific_merged_3.jsonl")}
    ge = {r["case_id"]: int(r["prediction"] == r["gold"]) for r in jl(d / "dsl_generic_merged_3.jsonl")}
    cat = {c: classify(r) for c, r in ini.items()}
    row = {"n": len(ini), "category_counts": {}}
    for k in set(cat.values()):
        row["category_counts"][k] = sum(v == k for v in cat.values())
    for k in ("alignment", "parse", "structural", "execution"):
        ids = [c for c in ini if cat[c] == k]
        if ids:
            row[k] = st.paired({c: sp[c] for c in ids}, {c: ge[c] for c in ids}, seed=20261011)
    A = row["category_counts"].get("alignment", 0); P = row["category_counts"].get("parse", 0)
    row["evaluable"] = A >= 15 and P >= 15
    if row["evaluable"]:
        pv[m] = row["alignment"]["mcnemar_exact_p"]
    res[m] = row
holm = st.holm(pv); ok_all = True; n_ev = 0
for m, row in res.items():
    if not row["evaluable"]:
        row["verdict"] = "NOT EVALUABLE"; continue
    n_ev += 1
    a = row["alignment"]; p = row["parse"]
    i_ok = a["difference"] > 0 and holm[m] < 0.05
    ii_ok = a["difference"] > p["difference"]
    row["holm_p_alignment"] = holm[m]; row["pass_i"] = i_ok; row["pass_ii"] = ii_ok
    row["verdict"] = "PASS" if (i_ok and ii_ok) else "FAIL"; ok_all &= (i_ok and ii_ok)
res["X2_verdict"] = "CONFIRMED" if (n_ev >= 2 and ok_all) else ("NOT CONFIRMED" if n_ev >= 2 else "NOT EVALUABLE")
json.dump(res, open(out_path, "w"), indent=1)
for m, row in res.items():
    if m == "X2_verdict": continue
    a = row.get("alignment"); p = row.get("parse")
    print(m, row["category_counts"], "evaluable", row["evaluable"])
    if a: print(f"  alignment n={a['n']} spec={a['candidate_accuracy']*100:.1f} gen={a['baseline_accuracy']*100:.1f} diff={a['difference']*100:+.1f} CI=[{a['ci95'][0]*100:+.1f},{a['ci95'][1]*100:+.1f}] p={a['mcnemar_exact_p']:.2g}")
    if p: print(f"  parse     n={p['n']} spec={p['candidate_accuracy']*100:.1f} gen={p['baseline_accuracy']*100:.1f} diff={p['difference']*100:+.1f} CI=[{p['ci95'][0]*100:+.1f},{p['ci95'][1]*100:+.1f}]")
    print("  verdict", row.get("verdict"))
print("X2:", res["X2_verdict"])
