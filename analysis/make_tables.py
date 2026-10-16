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


def tab_selmatch():
    """JIT-Repair minus each allocation policy, with the policy's winner replacing the output or passing the same
    selector (held-out BIRD, primary vote size). Source: analysis/selector_matched.csv (selector_matched.py)."""
    rows = rd(HERE / "selector_matched.csv")
    get = {(r["model"], r["comparator"]): r for r in rows if r["N"] == "3"}
    spec = [("RAND_U", "Random"), ("RAND_G", "Random, same selector"), ("CONF_sum_U", "Confidence"),
            ("CONF_sum_G", "Confidence, same selector"), ("LEN_q_U", "Length"), ("LEN_q_G", "Length, same selector"),
            ("VER_U", r"\jit{} without selector")]
    lines = [r"\begin{tabular}{@{}l r r r r l@{}}", r"\toprule",
             r"\jit{} minus & Llama-3.1-8B & Mistral-7B & Qwen3-8B & Qwen3-14B & Pooled [95\% CI] \\", r"\midrule"]
    for c, lab in spec:
        cells = [sg(100 * float(get[(m, c)]["diff"]), 2) for m in ("llama31_8b", "mistral7b", "qwen3_8b", "qwen3_14b")]
        pr = get[("POOLED", c)]
        pooled = f"{sg(100 * float(pr['diff']), 2)} [{sg(100 * float(pr['ci_lo']), 2)}, {sg(100 * float(pr['ci_hi']), 2)}]"
        if c == "VER_U":
            lines.append(r"\addlinespace")
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {pooled} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_selmatch.tex").write_text("\n".join(lines) + "\n")


def tab_existing():
    """JIT-Repair and its gated policies against existing ways of spending test-time compute, each comparison within
    one hardware stack. Sources: equal_budget.csv (self-consistency, SQL cells), archive T3/T6 and arm audit
    (Self-Refine, revision of every output), C1."""
    eb = rd(HERE / "equal_budget.csv")
    nc = sum(1 for r in eb if r["cell"] != "POOLED")
    P = {r["stat"]: r for r in eb if r["cell"] == "POOLED"}
    nN = {int(r["N_star"]) for r in eb if r["cell"] != "POOLED"}
    t3 = {(r["model"], r["row"].split(" = ")[0]): r for r in rd(REC / "tables/T3_like_for_like_cost.csv")}
    t6 = {(r["model"], r["comparator"]): r for r in rd(REC / "tables/T6_ablation.csv")}
    c1 = {(r["model"], r["set"], r["comparison"]): r for r in rd(REC / "tables/C1_comparisons.csv")}
    arch = jp.ARCHIVE / "drspider_oneuse"

    def d1(x, lo, hi):                       # difference in points with its 95% interval, one decimal
        return f"{sg(x)} [{sg(lo)}, {sg(hi)}]"

    def group(t):
        return [r"\addlinespace", r"\multicolumn{5}{@{}l}{\textit{" + t + r"}} \\"]

    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.5cm} r r l r@{}}", r"\toprule",
             r"Setting (our policy) & Existing & Ours & Ours $-$ existing [95\% CI] & Cost: existing / ours \\", r"\midrule"]
    lines += group("Self-consistency: a three-sample vote on every question")[1:]

    def pooled(key, lab, a, b, cost):
        s = P[key]
        e = [100 * float(s[k]) for k in ("estimate", "ci_lo", "ci_hi")]
        lines.append(f"{nc} SQL cells: {lab} & {100 * float(P[a]['estimate']):.2f} & {100 * float(P[b]['estimate']):.2f} & "
                     f"{sg(e[0], 2)} [{sg(e[1], 2)}, {sg(e[2], 2)}] & {cost} \\\\")
    pooled("diff_jit3_sc3", r"gain over greedy, points (\jit{}, vote of 3)", "gain_sc3", "gain_jit3",
           f"{float(P['calls_sc3']['estimate']):.2f} / {float(P['calls_jit3']['estimate']):.2f} calls")
    pooled("diff_broken_jit3_sc3", r"correct greedy outputs broken, \% (\jit{}, vote of 3)", "broken_sc3", "broken_jit3", "")
    pooled("diff_jitN_sc3", f"gain over greedy at no more calls, points (\\jit{{}}, vote of up to {max(nN)})", "gain_sc3",
           "gain_jitN", f"{float(P['calls_sc3']['estimate']):.2f} / {float(P['calls_jitN']['estimate']):.2f} calls")
    lines += group(r"Self-Refine: critique and rewrite every output\textsuperscript{a}")
    for mm, m in (("Llama-3.1-8B", "llama31_8b"), ("Mistral-7B", "mistral7b")):
        sr, ji, d = t3[(mm, "Self-Refine")], t3[(mm, "JIT-Repair")], t6[(mm, "self_refine")]
        lines.append(f"Dr.Spider, {mm} (\\jit{{}}, revise) & {f1(sr['accuracy'])} & {f1(ji['accuracy'])} & "
                     f"{d1(float(d['difference_pp']), float(d['cluster_ci95_lo_pp']), float(d['cluster_ci95_hi_pp']))} & "
                     f"{float(sr['model_calls_per_item']):.2f} / {float(ji['model_calls_per_item']):.2f} calls \\\\")
    lines += group(r"Revision of every output\textsuperscript{a}")
    for mm, m in (("Llama-3.1-8B", "llama31_8b"), ("Mistral-7B", "mistral7b")):
        al, ji = t6[(mm, "always_generic")], t3[(mm, "JIT-Repair")]
        au = json.load(open(arch / m / "arms_audit.json"))
        n_all = int(ji["items"]); n_rev = au["arms"]["subset_always_generic"]["selection_reasons"]["always_use_generic_repair"]
        lines.append(f"Dr.Spider, {mm} (\\jit{{}}, revise) & {f1(al['comparator_accuracy'])} & {f1(al['jit_accuracy'])} & "
                     f"{d1(float(al['difference_pp']), float(al['cluster_ci95_lo_pp']), float(al['cluster_ci95_hi_pp']))} & "
                     f"{(n_all + n_rev) / n_all:.2f} / {float(ji['model_calls_per_item']):.2f} calls \\\\")
    lines += group("The model's own thinking mode on ProofWriter")
    for mm, st, lab in (("Qwen3-14B", "CONF", "four-label"), ("Qwen3-14B", "D5", "depth-5"), ("Qwen3-8B", "D5", "depth-5")):
        r = c1[(mm, st, "JIT-think")]
        lines.append(f"{lab.capitalize()} set, {mm} (gated program) & {f1(r['baseline_acc'])} & {f1(r['jit_acc'])} & "
                     f"{d1(100 * float(r['diff']), 100 * float(r['ci_lo']), 100 * float(r['ci_hi']))} & "
                     f"{float(r['baseline_tokens_per_q']):,.0f} / {float(r['jit_tokens_per_q']):,.0f} tokens \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_existing.tex").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    tab_x7(); tab_c1(); tab_spend(); tab_selmatch(); tab_existing()
    for p in sorted(OUT.iterdir()):
        print("==", p.name); print(p.read_text())
