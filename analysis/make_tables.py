"""Generates the LaTeX tables of the manuscript from the records and the analysis outputs."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import csv, json, os, re
from pathlib import Path
REC = jp.RECORDS
HERE = jp.outputs("analysis")
OUT = jp.outputs("tables")
OUT.mkdir(parents=True, exist_ok=True)
LAB = {"llama31_8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B", "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B"}
BENCH = {"spider": "Spider", "bird": "BIRD Mini-Dev", "drspider_subset": "Dr.Spider", "birddev_rest": "BIRD dev (held out)"}


def rd(p):
    return list(csv.DictReader(open(p)))


def f1(x):
    return f"{100 * float(x):.1f}"


def f2(x):
    return f"{100 * float(x):.2f}"


def sg(x, nd=1):
    """signed number with a typographic minus"""
    t = f"{x:+.{nd}f}"
    return t.replace("-", "$-$")


def tab_x7():
    rows = rd(REC / "tables/X7_confirmation_allocation.csv")
    lines = [r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
             r"Model & $\rho$ & Greedy & Random & Confidence & Length & \jit{} \\",
             r"\midrule"]
    for m in ("llama31_8b", "mistral7b", "qwen3_8b", "qwen3_14b"):
        r = {x["comparator"]: x for x in rows if x["model"] == m and x["N"] == "3"}
        a = r["RAND_U"]
        lines.append(f"{LAB[m]} & {float(a['trigger_rate']):.2f} & {f1(a['greedy'])} & {f1(r['RAND_U']['comparator_acc'])} & "
                     f"{f1(r['CONF_sum_U']['comparator_acc'])} & {f1(r['LEN_q_U']['comparator_acc'])} & {f1(a['VER_G'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_x7.tex").write_text("\n".join(lines) + "\n")


def tab_c1():
    c1 = rd(REC / "tables/C1_comparisons.csv")
    g1 = json.load(open(REC / "tables/G1_llama_second_attempt.json"))
    name = {"JIT-think": "thinking", "JIT-cot": "step-by-step"}
    sets = {"CONF": "four-label", "D5": "depth-5"}
    lines = [r"\begin{tabular}{@{}l l l r r r r@{}}", r"\toprule",
             r"Model & Set & Baseline & \jit{} & Baseline & $\Delta$ [95\% CI] & Tokens\textsuperscript{b} \\",
             r"\midrule"]
    for r in c1:
        d, lo, hi = 100 * float(r["diff"]), 100 * float(r["ci_lo"]), 100 * float(r["ci_hi"])
        tok = f"{float(r['jit_tokens_per_q']):.0f} / {float(r['baseline_tokens_per_q']):.0f}"
        if r["model"] == "Llama-3.1-8B":
            ci = r"n.e.\textsuperscript{a}"
            r = dict(r, baseline_acc=g1["rule"]["baseline_accuracy"])
            tok = tok + r"\textsuperscript{c}"
        else:
            ci = f"{sg(d)} [{sg(lo)}, {sg(hi)}]"
        lines.append(f"{r['model']} & {sets[r['set']]} & {name[r['comparison']]} & {f1(r['jit_acc'])} & {f1(r['baseline_acc'])} & {ci} & {tok} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_c1.tex").write_text("\n".join(lines) + "\n")
    w = g1["worst_case_truncated_correct"]
    arms = json.load(open(REC / "tables/C1_T1_analysis.json"))["arms"]["llama31_8b|d5|cot"]["truncation_info"]
    second = int(re.search(r"max_tokens=(\d+)", open(Path(__file__).resolve().parents[1] / "experiments" / "llama_cot_second_attempt.py").read()).group(1))
    loop = 100 * g1["truncated_after_second_attempt"] / g1["n"]
    (OUT / "tab_c1_note.tex").write_text(
        f"\\textsuperscript{{a}}Not evaluable under the pre-registered rule: {loop:.1f}\\% of step-by-step outputs still loop at {second:,} tokens."
        f" Scored as wrong, $\\Delta$ = {sg(100*g1['rule']['difference'])} [{sg(100*g1['rule']['ci95'][0])}, {sg(100*g1['rule']['ci95'][1])}];"
        f" scored as correct, $\\Delta$ = {sg(100*w['difference'])} [{sg(100*w['ci95'][0])}, {sg(100*w['ci95'][1])}]."
        r" \textsuperscript{b}Generated tokens per question, \jit{} / baseline. \textsuperscript{c}Baseline tokens at the "
        f"{arms['declared_increase_to']:,}-token budget, before the declared second attempt at {second:,} tokens on still-truncated outputs." + "\n")


def tab_spend():
    th = {(r["cell"], r["N"]): r for r in rd(HERE / "allocation_theory_checks.csv")}
    bd = {r["cell"]: r for r in rd(REC / "tables/X7_boundary.csv")}
    c2 = rd(REC / "tables/C2_frontier.csv")
    x3b = json.load(open(REC / "results/x3b/x3b_summary.json"))["per_model"]
    x3c = json.load(open(REC / "results/x3b/x3c_qwen3_8b_summary.json"))["per_model"]
    x7 = json.load(open(REC / "tables/X7_confirmation_pooled.json"))["descriptive"]
    ch = {r["cell"]: r for r in rd(HERE / "allocation_lens_changes.csv") if r["N"] == "3"}
    order = [("spider", "llama31_8b"), ("spider", "mistral7b"), ("bird", "llama31_8b"), ("bird", "mistral7b"),
             ("drspider_subset", "llama31_8b"), ("drspider_subset", "mistral7b"), ("drspider_subset", "qwen3_8b"),
             ("birddev_rest", "llama31_8b"), ("birddev_rest", "mistral7b"), ("birddev_rest", "qwen3_8b"), ("birddev_rest", "qwen3_14b")]
    lines = [r"\begin{tabular}{@{}l l r r r r r r r@{}}", r"\toprule",
             r"Benchmark & Model & $\rho$ & Agree & Change & Greedy & \jit{} (revise) & \jit{} (vote) & Self-consistency \\",
             r"\midrule"]
    prev = None
    for b, m in order:
        cell = f"{b}/{m}"; t = th[(cell, "3")]
        if b in ("spider", "bird"):
            jr = [r for r in c2 if r["bench"] == b and r["model"] == m and r["method"] == "jit"][0]
            rev_acc, rev_calls = float(jr["accuracy"]), float(jr["mean_calls"])
        elif b == "drspider_subset":
            fr = (x3b.get(m) or x3c.get(m))["frontier"]["jit"]
            rev_acc, rev_calls = fr["accuracy"], fr["mean_calls"]
        else:
            rev_acc, rev_calls = x7[m]["JIT"]["acc"], x7[m]["JIT"]["calls"]
        if prev is not None and prev != b:
            lines.append(r"\addlinespace")
        prev = b
        bench = BENCH[b] if (b, m) == [o for o in order if o[0] == b][0] else ""
        lines.append(f"{bench} & {LAB[m]} & {float(t['rho']):.2f} & {float(bd[cell]['all3_agree']):.2f} & {float(ch[cell]['delta_flagged']):.2f} & {f1(t['greedy'])} & "
                     f"{f1(rev_acc)} ({rev_calls:.2f}) & {f1(t['gated_S'])} ({float(t['calls_gated_measured']):.2f}) & "
                     f"{f1(t['uniform_vote'])} ({float(t['calls_ungated_measured']):.2f}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_spend.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    tab_x7(); tab_c1(); tab_spend()
    for p in sorted(OUT.iterdir()):
        print("==", p.name); print(p.read_text())
