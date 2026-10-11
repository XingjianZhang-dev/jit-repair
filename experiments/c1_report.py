"""Tables/figures for C1 and the within-stack T1 (D5 + CONF) from c1_analyze output."""
from jit_repair import paths as jp  # noqa: E402
import csv, json, os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REC = jp.RECORDS
a = json.load(open(sys.argv[1]))
NICE = {"qwen3_14b": "Qwen3-14B", "qwen3_8b": "Qwen3-8B", "llama31_8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B"}
rows = []
for k, r in a["comparisons"].items():
    m, s, comp = k.split("|")
    row = {"model": NICE[m], "set": s.upper(), "comparison": comp, "verdict": r.get("verdict")}
    if "difference" in r:
        row.update({"n": r["n"], "chance": round(r["chance"], 4), "jit_acc": r["candidate_accuracy"], "baseline_acc": r["baseline_accuracy"],
                    "diff": r["difference"], "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1], "wins": r["candidate_only_correct"],
                    "losses": r["baseline_only_correct"], "mcnemar_p": r["mcnemar_exact_p"], "holm_p": r.get("holm_p"),
                    "baseline_truncation": r["baseline_truncation_rate"], "baseline_no_answer": r["baseline_no_answer"],
                    "jit_tokens_per_q": r["jit_mean_total_output_tokens"], "baseline_tokens_per_q": r["baseline_mean_output_tokens"],
                    "strict_parser_baseline_acc": r["strict_parser_sensitivity"]["baseline_accuracy"]})
    rows.append(row)
keys = sorted({k for r in rows for k in r}, key=lambda x: list(rows[0].keys()).index(x) if x in rows[0] else 99)
fields = ["model", "set", "comparison", "n", "chance", "jit_acc", "baseline_acc", "diff", "ci_lo", "ci_hi", "wins", "losses",
          "mcnemar_p", "holm_p", "baseline_truncation", "baseline_no_answer", "jit_tokens_per_q", "baseline_tokens_per_q",
          "strict_parser_baseline_acc", "verdict"]
with open(REC / "tables/C1_comparisons.csv", "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
L = [r"\begin{tabular}{lllrrrrrrl}", r"\toprule",
     r"Model & Set & Baseline & $n$ & JIT (\%) & Base (\%) & $\Delta$ (pp) & 95\% CI & Tok JIT/Base & Verdict \\", r"\midrule"]
for r in rows:
    if "diff" in r:
        L.append(f"{r['model']} & {r['set']} & {r['comparison'].replace('JIT-', '')} & {r['n']} & {100*r['jit_acc']:.1f} & {100*r['baseline_acc']:.1f} & "
                 f"{100*r['diff']:+.1f} & [{100*r['ci_lo']:+.1f}, {100*r['ci_hi']:+.1f}] & {r['jit_tokens_per_q']:.0f}/{r['baseline_tokens_per_q']:.0f} & {r['verdict'].split(' (')[0]} \\\\")
    else:
        L.append(f"{r['model']} & {r['set']} & {r['comparison'].replace('JIT-', '')} & -- & -- & -- & -- & -- & -- & {r['verdict']} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(REC / "tables/C1_comparisons.tex").write_text("\n".join(L) + "\n")
# T1 within-stack
t1rows = []
for k, t in a["t1"].items():
    m, s = k.split("|")
    row = {"model": NICE[m], "set": s.upper(), "n": t["n"], "chance": round(t["chance"], 4), **{f"acc_{x}": v for x, v in t["acc"].items()}}
    for c in ("formalisation", "retry", "diagnostic", "retry_brief", "diagnostic_round1", "specific_vs_initial"):
        if c in t:
            row[f"{c}_diff"] = t[c]["difference"]; row[f"{c}_ci_lo"] = t[c]["ci95"][0]; row[f"{c}_ci_hi"] = t[c]["ci95"][1]
            row[f"{c}_p"] = t[c]["mcnemar_exact_p"]
    t1rows.append(row)
f2 = sorted({k for r in t1rows for k in r}, key=lambda x: (not x.startswith(("model", "set", "n", "chance")), x))
with open(REC / "tables/T1_within_stack_decomposition.csv", "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=f2); w.writeheader(); w.writerows(t1rows)
fig, ax = plt.subplots(figsize=(9, 4.4))
import numpy as np
arms = [("direct", "direct label", "#BBBBBB"), ("initial", "initial program", "#4C72B0"), ("generic", "+ generic repair (x3)", "#55A868"),
        ("specific", "+ specific repair (x3) = JIT", "#C44E52"), ("cot", "step-by-step CoT", "#8172B2"), ("think", "Qwen3 thinking", "#CCB974")]
labels = [f"{r['model']}\n{r['set']}" for r in t1rows]
x = np.arange(len(t1rows)); w = 0.13
for k, (arm, lab, col) in enumerate(arms):
    vals = [100 * r.get(f"acc_{arm}", np.nan) for r in t1rows]
    ax.bar(x + (k - 2.5) * w, vals, w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 122)
ax.set_title("ProofWriter, all arms regenerated on one stack (A100/vLLM): where the accuracy comes from", fontsize=9)
ax.legend(fontsize=7, ncol=3, loc="upper center"); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
fig.savefig(REC / "figures/T1_within_stack_stacked.pdf"); fig.savefig(REC / "figures/T1_within_stack_stacked.png", dpi=150)
print("wrote C1 and T1 tables/figures;", a["C1_overall"])
