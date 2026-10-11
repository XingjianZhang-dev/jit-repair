"""Generates the supplementary LaTeX tables from the records and the analysis outputs."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import csv, json, os
from pathlib import Path
REC = jp.RECORDS
HERE = jp.outputs("analysis")
OUT = jp.outputs("supp_tables")
OUT.mkdir(parents=True, exist_ok=True)
LAB = {"llama31_8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B", "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "POOLED": "Pooled"}
BENCH = {"spider": "Spider", "bird": "BIRD Mini-Dev", "drspider_subset": "Dr.Spider", "birddev_rest": "BIRD dev (held out)"}


def rd(p):
    return list(csv.DictReader(open(p)))


def sg(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "$-$")


def pc(x, nd=1):
    return f"{100 * float(x):.{nd}f}"


def cellname(c):
    b, m = c.split("/")
    return f"{BENCH[b]} & {LAB[m]}"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")


def s2_checks():
    rows = [r for r in rd(HERE / "allocation_theory_checks.csv") if r["N"] == "3"]
    L = [r"\begin{tabular}{@{}l l r r r r r r r r@{}}", r"\toprule",
         r"Benchmark & Model & $\rho$ & $\bar\Delta_1$ & $\bar\Delta_0$ & $\widehat{\Cov}(G,\Delta)$ & $\rho(1{-}\rho)(\bar\Delta_1{-}\bar\Delta_0)$ & Bound & Gated$-$uniform & Calls \\",
         r"\midrule"]
    for r in rows:
        f = lambda k: 100 * float(r[k])
        L.append(f"{cellname(r['cell'])} & {float(r['rho']):.3f} & {sg(f('mu_U_flagged'))} & {sg(f('mu_U_passed'))} & {sg(f('cov_G_DU'), 3)} & "
                 f"{sg(100*float(r['rho'])*(1-float(r['rho']))*(float(r['mu_U_flagged'])-float(r['mu_U_passed'])), 3)} & {f('corollary_bound_U'):.2f} & "
                 f"{sg(f('gated_minus_uniform'))} & {float(r['calls_gated_measured']):.2f} / {float(r['calls_ungated_measured']):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s2_checks.tex", L)
    summ = json.load(open(HERE / "allocation_theory_checks.json"))
    lens = json.load(open(HERE / "allocation_lens.json"))
    import re
    (OUT / "s2_checks_summary.tex").write_text(
        f"Largest absolute residual over all {summ['rows']} cell--$N$ records: identity {max(summ['max_abs_residuals'][k] for k in ('T1_resid_U','T1_resid_S','T1b_resid_U','T1b_resid_S')):.1e}, "
        f"gated-versus-uniform {summ['max_abs_residuals']['T3_resid']:.1e}, selector decomposition {summ['max_abs_residuals']['T4_resid']:.1e}; "
        f"call formulas {summ['T6_max_abs_gated_calls_err']:.1e} (gated) and {summ['T6_max_abs_ungated_calls_err']:.1f} (uniform). "
        f"The outcome-change bound held in {summ['rows'] if summ['corollary_holds_all'] else 'NOT all'} of {summ['rows']} records, "
        f"the gold-free bounds of Corollary~2 in {lens['records'] if lens['gold_free_bound_holds_all'] and lens['uniform_bound_holds_all'] and lens['pi_le_delta_all'] else 'NOT all'} of {lens['records']}, and Proposition~1 had "
        f"{summ['T5_violations_total']} violations.\n")


def _fix_exp(path):
    import re
    t = path.read_text(); path.write_text(re.sub(r"(\d)e-(\d)", r"\1e$-$\2", t))


def s3_x7():
    rows = rd(REC / "tables/X7_confirmation_allocation.csv")
    comps = [("RAND_U", "random"), ("RAND_G", "random, with selector"), ("CONF_sum_U", "confidence (sum)"), ("CONF_mean_U", "confidence (mean)"),
             ("CONF_min_U", "confidence (min)"), ("LEN_sql_U", "length (query)"), ("LEN_q_U", "length (question)"), ("VER_U", "gate, no selector")]
    for N in ("2", "3", "5"):
        L = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule", r"Comparator & Llama-3.1-8B & Mistral-7B & Qwen3-8B & Qwen3-14B & Pooled \\", r"\midrule"]
        for c, lab in comps:
            cells = []
            for m in ("llama31_8b", "mistral7b", "qwen3_8b", "qwen3_14b", "POOLED"):
                r = [x for x in rows if x["model"] == m and x["N"] == N and x["comparator"] == c][0]
                cells.append(f"{sg(100*float(r['VER_G_minus']))} [{sg(100*float(r['ci_lo']))}, {sg(100*float(r['ci_hi']))}]")
            L.append(lab + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(f"s3_x7_N{N}.tex", L)


SETN = {"CONF": "four-label", "D5": "depth-5"}


def s4_c2():
    rows = rd(REC / "tables/C2_frontier.csv")
    order = ["greedy", "jit", "filter@1", "filter@2", "filter@3", "filter@5", "filter@8", "vote@1", "vote@2", "vote@3", "vote@5", "vote@8"]
    name = {"greedy": "greedy", "jit": "revise (gated)"}
    L = [r"\begin{tabular}{@{}l r r r r r r r r@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Spider, Llama} & \multicolumn{2}{c}{Spider, Mistral} & \multicolumn{2}{c}{BIRD, Llama} & \multicolumn{2}{c}{BIRD, Mistral} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Arm & Acc. & Calls & Acc. & Calls & Acc. & Calls & Acc. & Calls \\", r"\midrule"]
    for a in order:
        cells = []
        for b, m in (("spider", "llama31_8b"), ("spider", "mistral7b"), ("bird", "llama31_8b"), ("bird", "mistral7b")):
            r = [x for x in rows if x["bench"] == b and x["model"] == m and x["method"] == a][0]
            cells += [pc(r["accuracy"], 2), f"{float(r['mean_calls']):.2f}"]
        L.append(name.get(a, a.replace("@", "@")) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_c2.tex", L)
    # generated tokens per question: revision of flagged outputs vs a three-sample vote on every question
    L = [r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule", r" & Spider, Llama & Spider, Mistral & BIRD, Llama & BIRD, Mistral \\", r"\midrule"]
    tok = {}
    for a in ("jit", "vote@3"):
        cells = []
        for b, m in (("spider", "llama31_8b"), ("spider", "mistral7b"), ("bird", "llama31_8b"), ("bird", "mistral7b")):
            r = [x for x in rows if x["bench"] == b and x["model"] == m and x["method"] == a][0]
            tok[(a, b, m)] = float(r["mean_generated_tokens"]); cells.append(f"{tok[(a, b, m)]:.0f}")
        L.append(name.get(a, a) + " & " + " & ".join(cells) + r" \\")
    L.append(r"vote@3 / revise & " + " & ".join(f"{tok[('vote@3', b, m)] / tok[('jit', b, m)]:.2f}" for b, m in (("spider", "llama31_8b"), ("spider", "mistral7b"), ("bird", "llama31_8b"), ("bird", "mistral7b"))) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_c2_tokens.tex", L)


def s4_drspider():
    x3b = json.load(open(REC / "results/x3b/x3b_summary.json"))["per_model"]
    x3c = json.load(open(REC / "results/x3b/x3c_qwen3_8b_summary.json"))["per_model"]
    fr = {"llama31_8b": x3b["llama31_8b"]["frontier"], "mistral7b": x3b["mistral7b"]["frontier"], "qwen3_8b": x3c["qwen3_8b"]["frontier"]}
    order = ["greedy", "jit"] + [f"vote@{n}" for n in (1, 2, 3, 5, 8)] + [f"gated_vote@{n}" for n in (1, 2, 3, 5, 8)]
    name = {"greedy": "greedy", "jit": "revise (gated)"}
    L = [r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Llama-3.1-8B} & \multicolumn{2}{c}{Mistral-7B} & \multicolumn{2}{c}{Qwen3-8B} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r"Arm & Acc. & Calls & Acc. & Calls & Acc. & Calls \\", r"\midrule"]
    for a in order:
        cells = []
        for m in ("llama31_8b", "mistral7b", "qwen3_8b"):
            v = fr[m][a]; cells += [pc(v["accuracy"], 2), f"{v['mean_calls']:.2f}"]
        lab = name.get(a, a.replace("gated_vote", "gated vote"))
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_drspider.tex", L)


def s4_x3():
    d = json.load(open(REC / "results/x3/x3_confirmation_summary.json"))
    L = [r"\begin{tabular}{@{}l l r r r r r@{}}", r"\toprule", r"Benchmark & Model & $n$ & Self-consistency & + gated revision & $\Delta$ [95\% CI] & Fixes / breaks \\", r"\midrule"]
    for key in ("spider|llama31_8b|N3", "spider|mistral7b|N3", "bird|llama31_8b|N3", "bird|mistral7b|N3"):
        r = d[key]; b, m, _ = key.split("|")
        L.append(f"{'Spider (odd items)' if b == 'spider' else 'BIRD Mini-Dev'} & {LAB[m]} & {r['n']:,} & {pc(r['baseline_accuracy'],2)} & {pc(r['candidate_accuracy'],2)} & "
                 f"{sg(100*r['difference'])} [{sg(100*r['ci95'][0])}, {sg(100*r['ci95'][1])}] & {r['candidate_only_correct']} / {r['baseline_only_correct']} \\\\")
    p = d["pooled_N3"]
    L.append(f"Pooled & & {p['n']:,} & & & {sg(100*p['difference'])} [{sg(100*p['ci95'][0])}, {sg(100*p['ci95'][1])}] & {p['candidate_only_correct']} / {p['baseline_only_correct']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_x3.tex", L)


def s4_mechanism():
    rows = [r for r in rd(HERE / "allocation_theory_checks.csv") if r["N"] == "3"]
    L = [r"\begin{tabular}{@{}l l r r r r r r@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Flagged, no selector} & \multicolumn{2}{c}{Flagged, selector} & \multicolumn{2}{c}{Passed, no selector} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}", r"Benchmark & Model & Fixes & Breaks & Fixes & Breaks & Fixes & Breaks \\", r"\midrule"]
    tot = [0] * 6
    for r in rows:
        v = [int(r[k]) for k in ("flagged_fixed_noselector", "flagged_broken_noselector", "flagged_fixed_selector", "flagged_broken_selector", "unflagged_wrong_fixed", "unflagged_correct_broken")]
        tot = [a + b for a, b in zip(tot, v)]
        L.append(f"{cellname(r['cell'])} & " + " & ".join(str(x) for x in v) + r" \\")
    L += [r"\midrule", "Total & & " + " & ".join(str(x) for x in tot) + r" \\", r"\bottomrule", r"\end{tabular}"]
    write("s4_mechanism.tex", L)


def s4_c1():
    c1 = rd(REC / "tables/C1_comparisons.csv")
    L = [r"\begin{tabular}{@{}l l l r r r r r r@{}}", r"\toprule",
         r"Model & Set & Baseline & $n$ & \jit{} & Baseline & Strict parser & Holm $p$ & Verdict \\", r"\midrule"]
    for r in c1:
        hp = r["holm_p"]
        hp = "--" if not hp else f"{float(hp):.1e}".replace("e-", "e$-$")
        L.append(f"{r['model']} & {SETN[r['set']]} & {r['comparison'].replace('JIT-think','thinking').replace('JIT-cot','step-by-step')} & {r['n']} & {pc(r['jit_acc'])} & {pc(r['baseline_acc'])} & "
                 f"{pc(r['strict_parser_baseline_acc'])} & {hp} & {r['verdict'].split(' (')[0]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_c1.tex", L)
    t1 = rd(REC / "tables/T1_within_stack_decomposition.csv")
    L = [r"\begin{tabular}{@{}l l r r r r r r@{}}", r"\toprule",
         r"Model & Set & Direct & First program & Generic revision & Diagnostic revision & Step-by-step & Thinking \\", r"\midrule"]
    for r in t1:
        th = pc(r["acc_think"]) if r["acc_think"] else "--"
        L.append(f"{r['model']} & {SETN[r['set']]} & {pc(r['acc_direct'])} & {pc(r['acc_initial'])} & {pc(r['acc_generic'])} & {pc(r['acc_specific'])} & {pc(r['acc_cot'])} & {th} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_t1.tex", L)


def s4_archive():
    t6 = rd(REC / "tables/T6_ablation.csv")
    L = [r"\begin{tabular}{@{}l r r r@{}}", r"\toprule", r"Gated revision minus & Llama-3.1-8B & Mistral-7B & \\", r"\midrule"]
    L = [r"\begin{tabular}{@{}l r r@{}}", r"\toprule", r"Gated revision minus \ldots & Llama-3.1-8B & Mistral-7B \\", r"\midrule"]
    for comp in ("direct", "random_gate", "always_generic", "no_selector", "execution_only", "self_refine"):
        cells = []
        for m in ("Llama-3.1-8B", "Mistral-7B"):
            r = [x for x in t6 if x["model"] == m and x["comparator"] == comp][0]
            cells.append(f"{sg(float(r['difference_pp']))} [{sg(float(r['cluster_ci95_lo_pp']))}, {sg(float(r['cluster_ci95_hi_pp']))}]")
        lab = {"direct": "greedy, no extra compute", "random_gate": "random gate, same revision budget", "always_generic": "revise every output",
               "no_selector": "gate, no selector", "execution_only": "execution-only gate", "self_refine": "Self-Refine"}[comp]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_t6.tex", L)
    t3 = rd(REC / "tables/T3_like_for_like_cost.csv")
    L = [r"\begin{tabular}{@{}l l r r r r@{}}", r"\toprule", r"Model & Arm & Calls/q & Tokens/q & Seconds/q & Accuracy \\", r"\midrule"]
    for r in t3:
        if r["kind"] != "value" or "stage only" in r["row"]:
            continue
        arm = {"Direct": "greedy", "JIT-Repair": "gated revision", "Self-Refine": "Self-Refine"}[r["row"].split(" =")[0].split(" (")[0]]
        L.append(f"{r['model']} & {arm} & {float(r['model_calls_per_item']):.2f} & {float(r['output_tokens_per_item']):.1f} & {float(r['generation_seconds_per_item']):.3f} & {pc(r['accuracy'],2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_t3.tex", L)
    t9 = json.load(open(REC / "results/t9/t9_summary.json"))
    L = [r"\begin{tabular}{@{}l r r r@{}}", r"\toprule", r"Model & $n$ (clusters) & Test-suite gain & Single-database gain \\", r"\midrule"]
    for m in ("llama31_8b", "mistral7b"):
        a = t9[m]["testsuite_full_minus_direct"]; b = t9[m]["paper_single_db_full_minus_direct_same_items"]
        L.append(f"{LAB[m]} & {a['n']:,} ({t9[m]['clusters']:,}) & {sg(100*a['difference'])} [{sg(100*a['ci95'][0])}, {sg(100*a['ci95'][1])}] & "
                 f"{sg(100*b['difference'])} [{sg(100*b['ci95'][0])}, {sg(100*b['ci95'][1])}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_t9.tex", L)


def s4_g2g3():
    g2 = rd(REC / "tables/G2_allocation_controls.csv")
    L = [r"\begin{tabular}{@{}l l r r r r@{}}", r"\toprule",
         r"Benchmark & Model & $N$ & $\rho$ (execution only) & Execution-only gate $-$ random & Full gate $-$ execution-only gate \\", r"\midrule"]
    for r in g2:
        if r["comparator"].startswith("exec_only_gate vs random"):
            c = [x for x in g2 if x["cell"] == r["cell"] and x["N"] == r["N"] and x["comparator"].startswith("exec_only_gate (critic")][0]
            L.append(f"{cellname(r['cell'])} & {r['N']} & {float(r['trigger_rate']):.3f} & "
                     f"{sg(100*float(r['gated_minus_comparator']))} [{sg(100*float(r['ci_lo']))}, {sg(100*float(r['ci_hi']))}] & "
                     f"{sg(100*float(c['gated_minus_comparator']))} [{sg(100*float(c['ci_lo']))}, {sg(100*float(c['ci_hi']))}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_g2.tex", L)
    g3 = rd(REC / "tables/G3_prop1_checks.csv")
    L = [r"\begin{tabular}{@{}l l l r r r r@{}}", r"\toprule",
         r"Benchmark & Model & Policy & (i) & (ii) & (iii) & With selector $-$ without \\", r"\midrule"]
    for r in g3:
        if r["method"] == "JIT (within-stack)" or r["method"].startswith("gated_vote@"):
            pol = "revise" if r["method"].startswith("JIT") else r["method"].replace("_", " ")
            gg = "--" if not r["guarded_minus_gated"] else f"{sg(100*float(r['guarded_minus_gated']))} [{sg(100*float(r['gg_ci_lo']))}, {sg(100*float(r['gg_ci_hi']))}]"
            L.append(f"{cellname(r['cell'])} & {pol} & {r['viol_i_valid_to_invalid']} & {r['viol_ii_unflagged_changed']} & {r['viol_iii_critic_not_decreased']}/{r['valid_to_valid_changes']} & {gg} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s4_g3.tex", L)


def s5_ledger():
    rows = rd(REC / "LEDGER.csv")
    keep = ["C1", "C1-L", "C1-L2", "C2", "X1", "X2", "X3", "X3b", "X3c", "X7-P1", "X7-P2"]
    L = [r"\begin{longtable}{@{}p{0.07\linewidth} >{\raggedright\arraybackslash}p{0.30\linewidth} >{\raggedright\arraybackslash}p{0.30\linewidth} >{\raggedright\arraybackslash}p{0.14\linewidth} >{\raggedright\arraybackslash}p{0.11\linewidth}@{}}", r"\toprule",
         r"ID & Prediction & Rule fixed in advance & Data & Verdict \\", r"\midrule", r"\endhead"]
    esc = lambda t: t.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("≥", r"$\ge$").replace("≤", r"$\le$").replace("−", r"$-$").replace("Δ", r"$\Delta$").replace(">", r"$>$").replace("<", r"$<$")
    for k in keep:
        r = [x for x in rows if x["id"] == k][0]
        L.append(f"{k} & {esc(r['test'])} & {esc(r['rule_fixed_in_advance'])} & {esc(r['set_and_n'])} & {esc(r['verdict'])} \\\\")
    L += [r"\bottomrule", r"\end{longtable}"]
    write("s5_ledger.tex", L)
    L = [r"\begin{tabular}{@{}p{0.07\linewidth} >{\raggedright\arraybackslash}p{0.36\linewidth} >{\raggedright\arraybackslash}p{0.27\linewidth} >{\raggedright\arraybackslash}p{0.20\linewidth}@{}}", r"\toprule", r"ID & Check & Result & Verdict \\", r"\midrule"]
    for k in ("G0", "G1a", "G1b", "G2", "G3", "G4", "G5", "G6"):
        r = [x for x in rows if x["id"] == k][0]
        L.append(f"{k} & {esc(r['test'])} & {esc(r['result'])} & {esc(r['verdict'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("s6_gates.tex", L)


if __name__ == "__main__":
    s2_checks(); _fix_exp(OUT / "s2_checks_summary.tex"); s3_x7(); s4_c2(); s4_drspider(); s4_x3(); s4_mechanism(); s4_c1(); s4_archive(); s4_g2g3(); s5_ledger()
    print(sorted(p.name for p in OUT.iterdir()))
