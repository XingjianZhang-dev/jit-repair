"""T6: main-text ablation table, 2 models x 6 comparators, JIT-Repair (subset_full) minus comparator on the
1,556-item Dr.Spider subset (780 q_id_spider_dev clusters). Frozen numbers from analysis.json
(subset_comparisons.full_vs_*; cluster bootstrap, 50,000 samples, stored per-contrast seed); every contrast is
re-derived from the cases files with paired_stats.paired(..., clusters=safe.jsonl q_id_spider_dev, seed=stored,
samples=50,000) and compared. Holm is within this 12-test family; cluster p is the T4 sign-flip test."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from sql_archive_common import (DR, MODELS, MODEL_LABEL, COMPARATORS, TABLES, paired_stats, load_json, safe_rows,
                    correctness, sign_flip_p, write_csv, fmt_p)

LABEL = {"direct": "Direct (no repair)", "self_refine": "Self-Refine (feedback + iterate)",
         "always_generic": "Always repair (no gate)", "execution_only": "Execution-only gate",
         "no_selector": "Gate, no selector (accept every repair)",
         "random_gate": "Random gate (same repair budget)"}


def sg(x: float) -> str:
    return "0.00" if round(x, 2) == 0 else f"{x:+.2f}"


def main():
    _, clusters, subset = safe_rows()
    rows = []
    for m in MODELS:
        an = load_json(DR / m / "analysis.json")
        assert an["bootstrap_samples"] == 50_000
        cand = correctness(DR / m / "cases" / "subset_full.cases.jsonl")
        case_q = {cid: int(r) for cid, r in
                  ((c, x["q_id_spider_dev"]) for c, x in
                   __import__("sql_archive_common").keyed(DR / m / "cases" / "subset_full.cases.jsonl").items())}
        assert all(case_q[c] == clusters[c] for c in case_q), "cases cluster != safe.jsonl cluster"
        for comp_name in COMPARATORS:
            s = an["subset_comparisons"][f"full_vs_{comp_name}"]
            comp = correctness(DR / m / "cases" / f"subset_{comp_name}.cases.jsonl")
            assert set(comp) == set(cand) == subset
            st = paired_stats.paired(cand, comp, seed=int(s["seed"]), samples=int(s["bootstrap_samples"]),
                                clusters={c: clusters[c] for c in cand})
            lo, hi = s["cluster_bootstrap_difference_ci95"]
            match_exact = (st["ci95"][0] == lo and st["ci95"][1] == hi)
            max_abs = max(abs(st["ci95"][0] - lo), abs(st["ci95"][1] - hi))
            assert (st["candidate_only_correct"], st["baseline_only_correct"]) == \
                (s["candidate_only_correct"], s["direct_only_correct"])
            assert abs(st["difference"] - s["accuracy_difference"]) < 1e-12
            perm = sign_flip_p(cand, comp, clusters)
            rows.append({
                "model": MODEL_LABEL[m], "comparator": comp_name, "comparator_label": LABEL[comp_name],
                "n": s["records"], "clusters": s["original_question_clusters"],
                "jit_correct": sum(cand.values()), "jit_accuracy": s["candidate_accuracy"],
                "comparator_correct": sum(comp.values()), "comparator_accuracy": s["direct_accuracy"],
                "difference_pp": 100 * s["accuracy_difference"],
                "cluster_ci95_lo_pp": 100 * lo, "cluster_ci95_hi_pp": 100 * hi,
                "jit_only_correct": s["candidate_only_correct"],
                "comparator_only_correct": s["direct_only_correct"],
                "mcnemar_exact_p": s["mcnemar_exact_p"], "cluster_perm_p": perm["cluster_perm_p"],
                "bootstrap_seed": s["seed"], "bootstrap_samples": s["bootstrap_samples"],
                "recomputed_ci95_lo_pp": 100 * st["ci95"][0], "recomputed_ci95_hi_pp": 100 * st["ci95"][1],
                "recompute_exact_match": match_exact, "recompute_max_abs_diff_pp": 100 * max_abs,
            })
    h = paired_stats.holm({(r["model"], r["comparator"]): r["mcnemar_exact_p"] for r in rows})
    for r in rows:
        r["holm_p_12"] = h[(r["model"], r["comparator"])]
    fields = ["model", "comparator", "comparator_label", "n", "clusters", "jit_correct", "jit_accuracy",
              "comparator_correct", "comparator_accuracy", "difference_pp", "cluster_ci95_lo_pp",
              "cluster_ci95_hi_pp", "jit_only_correct", "comparator_only_correct", "mcnemar_exact_p",
              "holm_p_12", "cluster_perm_p", "bootstrap_seed", "bootstrap_samples", "recomputed_ci95_lo_pp",
              "recomputed_ci95_hi_pp", "recompute_exact_match", "recompute_max_abs_diff_pp"]
    write_csv(TABLES / "T6_ablation.csv", rows, fields)
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Ablations on the Dr.Spider subset ($n=1{,}556$ perturbations from 780 original questions; same items for every arm). "
         r"$\Delta$ is JIT-Repair minus the comparator in accuracy points; the 95\% CI is a cluster bootstrap over original questions (50{,}000 resamples). "
         r"$N_{01}$/$N_{10}$: items only JIT-Repair / only the comparator gets right. $p$: exact McNemar; Holm over these 12 contrasts.}",
         r"\label{tab:ablation}",
         r"\begin{tabular}{llrrrrrrr}\toprule",
         r"Model & Comparator & Comp.\ acc. & JIT acc. & $\Delta$ (pp) & 95\% CI (pp) & $N_{01}/N_{10}$ & McNemar $p$ & Holm $p$ \\ \midrule"]
    for i, r in enumerate(rows):
        model = r["model"] if r["comparator"] == "direct" else ""
        L.append(f"{model} & {r['comparator_label']} & {100*r['comparator_accuracy']:.2f} & "
                 f"{100*r['jit_accuracy']:.2f} & {r['difference_pp']:+.2f} & "
                 f"[{sg(r['cluster_ci95_lo_pp'])}, {sg(r['cluster_ci95_hi_pp'])}] & "
                 f"{r['jit_only_correct']}/{r['comparator_only_correct']} & {fmt_p(r['mcnemar_exact_p'])} & "
                 f"{fmt_p(r['holm_p_12'])} \\\\")
        if i == 5:
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    (TABLES / "T6_ablation.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['model']:13} {r['comparator']:15} n={r['n']} comp={r['comparator_correct']}/{r['n']} "
              f"jit={r['jit_correct']}/{r['n']} d={r['difference_pp']:+.2f} CI=[{r['cluster_ci95_lo_pp']:+.3f},{r['cluster_ci95_hi_pp']:+.3f}] "
              f"rec=[{r['recomputed_ci95_lo_pp']:+.3f},{r['recomputed_ci95_hi_pp']:+.3f}] exact={r['recompute_exact_match']} "
              f"maxdiff={r['recompute_max_abs_diff_pp']:.2e} {r['jit_only_correct']}/{r['comparator_only_correct']} "
              f"p={r['mcnemar_exact_p']:.3g} holm={r['holm_p_12']:.3g} perm={r['cluster_perm_p']:.3g}")


if __name__ == "__main__":
    main()
