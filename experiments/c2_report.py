"""C2 verdict, frontier figures and tables from the four cell summaries (local, CPU)."""
from jit_repair import paths as jp  # noqa: E402
import json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
REC = jp.RECORDS
E = jp.ARCHIVE / "evidence"
RES = REC / "results/c2"
CELLS = [("spider", "llama31_8b"), ("spider", "mistral7b"), ("bird", "llama31_8b"), ("bird", "mistral7b")]
FROZEN = {("spider", "llama31_8b"): E / "spider_generic_forward_canonical_v2_full",
          ("spider", "mistral7b"): E / "spider_generic_forward_cross_model_public_test_v1_full/mistral7b",
          ("bird", "llama31_8b"): E / "bird_minidev_generic_forward_two_model_v1_full/results/llama31_8b",
          ("bird", "mistral7b"): E / "bird_minidev_generic_forward_two_model_v1_full/results/mistral7b"}
NICE = {"llama31_8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B", "spider": "Spider test", "bird": "BIRD Mini-Dev"}
def jl(p): return [json.loads(l) for l in open(p)]

def frozen_point(b, m):
    d = FROZEN[(b, m)]
    ini = jl(d / "initial/initial.jsonl"); rep = {r["case_id"]: r for r in jl(d / "candidate/repair_1.jsonl")}
    ev = jl(d / "evaluation/candidate.cases.jsonl"); n = len(ini)
    return {"accuracy": sum(r["correct"] for r in ev) / n,
            "mean_calls": (n + len(rep)) / n,
            "mean_generated_tokens": (sum(r["output_tokens"] for r in ini) + sum(r["output_tokens"] for r in rep.values())) / n,
            "greedy_accuracy": sum(r["correct"] for r in jl(d / "evaluation/initial.cases.jsonl")) / n,
            "greedy_tokens": sum(r["output_tokens"] for r in ini) / n}

rows, verdict_rows = [], []
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, (b, m) in zip(axes, CELLS):
    s = json.load(open(RES / b / m / "c2_summary.json")); f = s["frontier"]
    fz = frozen_point(b, m)
    for meth, v in f.items():
        rows.append({"bench": b, "model": m, "n": s["n"], "method": meth, "accuracy": v["accuracy"],
                     "mean_calls": v["mean_calls"], "mean_generated_tokens": v["mean_generated_tokens"],
                     "jit_minus_method": None if meth == "jit" else s["stats"][f"jit_minus_{meth}"]["difference"],
                     "ci_lo": None if meth == "jit" else s["stats"][f"jit_minus_{meth}"]["ci95"][0],
                     "ci_hi": None if meth == "jit" else s["stats"][f"jit_minus_{meth}"]["ci95"][1],
                     "mcnemar_p": None if meth == "jit" else s["stats"][f"jit_minus_{meth}"]["mcnemar_exact_p"]})
    rows.append({"bench": b, "model": m, "n": s["n"], "method": "jit_FROZEN_ARCHIVE(descriptive)", "accuracy": fz["accuracy"],
                 "mean_calls": fz["mean_calls"], "mean_generated_tokens": fz["mean_generated_tokens"],
                 "jit_minus_method": None, "ci_lo": None, "ci_hi": None, "mcnemar_p": None})
    st2 = s["stats"]["jit_minus_vote@2"]
    verdict_rows.append({"cell": f"{b}/{m}", "n": s["n"], "jit": f["jit"]["accuracy"], "vote2": f["vote@2"]["accuracy"],
                         "diff": st2["difference"], "ci": st2["ci95"], "p": st2["mcnemar_exact_p"],
                         "point_ge_0": st2["difference"] >= 0, "vote2_sig_better": st2["ci95"][1] < 0,
                         "timeout_rate": s["timeout_rate"], "evaluable": s["timeout_rate"] <= 0.01})
    for sel, mk, col in (("vote", "o-", "#4C72B0"), ("filter", "s--", "#55A868")):
        xs = [f[f"{sel}@{N}"]["mean_generated_tokens"] for N in (1, 2, 3, 5, 8)]
        ys = [100 * f[f"{sel}@{N}"]["accuracy"] for N in (1, 2, 3, 5, 8)]
        ax.plot(xs, ys, mk, color=col, label=f"exec-{sel}@N (N=1,2,3,5,8)", ms=4)
        for N, x, y in zip((1, 2, 3, 5, 8), xs, ys):
            if sel == "vote": ax.annotate(str(N), (x, y), textcoords="offset points", xytext=(3, -9), fontsize=7, color=col)
    ax.plot(f["greedy"]["mean_generated_tokens"], 100 * f["greedy"]["accuracy"], "kx", ms=7, label="greedy (direct)")
    ax.plot(f["jit"]["mean_generated_tokens"], 100 * f["jit"]["accuracy"], "*", color="#C44E52", ms=13, label="JIT-Repair")
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel("mean generated tokens / question (log)"); ax.set_ylabel("execution accuracy (%)")
    ax.set_title(f"{NICE[b]}, {NICE[m]} (n={s['n']})", fontsize=10); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, loc="lower right")
fig.suptitle("C2: accuracy–cost frontier, all arms regenerated on one stack (vLLM, A100); JIT-Repair marked", fontsize=11)
fig.tight_layout(); (REC / "figures").mkdir(exist_ok=True)
fig.savefig(REC / "figures/C2_frontier.pdf"); fig.savefig(REC / "figures/C2_frontier.png", dpi=150)
import csv
with open(REC / "tables/C2_frontier.csv", "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cond1 = sum(r["point_ge_0"] for r in verdict_rows if r["evaluable"]) >= 3
cond2 = not any(r["vote2_sig_better"] for r in verdict_rows if r["evaluable"])
evaluable = all(r["evaluable"] for r in verdict_rows)
verdict = ("CONFIRMED" if (cond1 and cond2) else "NOT CONFIRMED") if evaluable else "NOT EVALUABLE (timeouts)"
json.dump({"cells": verdict_rows, "cond_ge3_cells_point_ge0": cond1, "cond_no_cell_vote2_sig_better": cond2,
           "C2_verdict": verdict}, open(REC / "tables/C2_verdict.json", "w"), indent=1)
# LaTeX table: main operating points
lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Cell & Method & Acc. (\%) & Calls/q & Gen. tok/q & JIT$-$method (pp) & 95\% CI & $p$ \\", r"\midrule"]
for r in rows:
    if r["method"] in ("greedy", "jit", "filter@8", "vote@2", "vote@3", "vote@5", "vote@8", "jit_FROZEN_ARCHIVE(descriptive)"):
        ci = "" if r["ci_lo"] is None else f"[{100*r['ci_lo']:+.2f}, {100*r['ci_hi']:+.2f}]"
        d = "" if r["jit_minus_method"] is None else f"{100*r['jit_minus_method']:+.2f}"
        p = "" if r["mcnemar_p"] is None else f"{r['mcnemar_p']:.2g}"
        meth = r["method"].replace("_FROZEN_ARCHIVE(descriptive)", " (frozen archive)").replace("@", "@")
        lines.append(f"{NICE[r['bench']]} / {NICE[r['model']]} & {meth} & {100*r['accuracy']:.2f} & {r['mean_calls']:.2f} & {r['mean_generated_tokens']:.1f} & {d} & {ci} & {p} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(REC / "tables/C2_frontier.tex").write_text("\n".join(lines) + "\n")
for r in verdict_rows:
    print(r["cell"], "n", r["n"], f"JIT {100*r['jit']:.2f} vote@2 {100*r['vote2']:.2f} diff {100*r['diff']:+.2f} CI [{100*r['ci'][0]:+.2f},{100*r['ci'][1]:+.2f}] p={r['p']:.3g} timeout {100*r['timeout_rate']:.2f}%")
print("C2:", verdict, "cond1", cond1, "cond2", cond2)
