"""Check of paired_stats on a known case: reproduce the frozen Qwen3-14B CONF paired statistics
(direct vs final JIT) exactly from the archived per-item records."""
from jit_repair import paths as jp  # noqa: E402
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import paired_stats as st  # noqa
E = jp.ARCHIVE / "evidence/qwen3_14b_confirmation"
LAB = {"true", "false", "both", "unknown"}
def jl(p): return [json.loads(l) for l in open(p)]
d = {r["case_id"]: int(r["prediction"] in LAB and r["prediction"] == r["gold"]) for r in jl(E / "direct.jsonl")}
c = {r["case_id"]: int(r["prediction"] in LAB and r["prediction"] == r["gold"]) for r in jl(E / "dsl_merged_3.jsonl")}
res = st.paired(c, d, seed=20260820, samples=10000)
fz = json.load(open(E / "paired_statistics.json"))
print(res)
assert abs(res["difference"] - fz["accuracy_difference"]) < 1e-12
assert all(abs(x - y) < 1e-12 for x, y in zip(res["ci95"], fz["accuracy_difference_ci95"])), (res["ci95"], fz["accuracy_difference_ci95"])
assert res["candidate_only_correct"] == fz["candidate_only_correct"] and res["baseline_only_correct"] == fz["direct_only_correct"]
assert abs(res["mcnemar_exact_p"] - fz["mcnemar_exact_p"]) / fz["mcnemar_exact_p"] < 1e-9
h = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
assert abs(h["a"] - 0.03) < 1e-12 and abs(h["c"] - 0.06) < 1e-12 and abs(h["b"] - 0.06) < 1e-12
print("ALL_PAIRED_STATS_TESTS_PASS")
