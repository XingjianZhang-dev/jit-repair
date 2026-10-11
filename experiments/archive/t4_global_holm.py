"""T4: global Holm over the 14 principal archive cells + 12 Dr.Spider ablation contrasts, plus a
cluster-aware sign-flip permutation p for every Dr.Spider row.

Raw p-values are the frozen exact McNemar p-values stored in each cell's source JSON (cells and sources
in PRINCIPAL_CELLS below); each is re-derived from the stored
discordant counts as a check. Dr.Spider McNemar p ignores the q_id_spider_dev clustering; the permutation
test flips the sign of each cluster's summed (candidate - comparator) correctness difference
(20,000 draws, numpy default_rng(20261011), fresh per contrast, two-sided, p=(1+#|T*|>=|T|)/(1+B))."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402

import sys
sys.dont_write_bytecode = True
from sql_archive_common import (DR, MODELS, MODEL_LABEL, COMPARATORS, TABLES, PERM_B, PERM_SEED, paired_stats,
                    load_json, safe_rows, correctness, sign_flip_p, write_csv, tex_escape, fmt_p)

P_PATH = {  # raw exact McNemar p in each source JSON (index into PAIRED_CELLS)
    0: "mcnemar_exact_p",
    1: "mcnemar_exact_p",
    2: "metrics.paired.full_context_direct.mcnemar_exact_p",
    3: "mcnemar_exact_p",
    4: "result.qwen3_14b_mcnemar_exact_p",
    5: "llama31_8b_confirmation.mcnemar_exact_p",
    6: "result.mistral7b.mcnemar_exact_p",
    7: "public_test_confirmation.mcnemar_exact_p",
    8: "mistral7b.mcnemar_exact_p",
    9: "qwen3_8b.mcnemar_exact_p",
    10: "llama31_8b.mcnemar_exact_p",
    11: "mistral7b.mcnemar_exact_p",
    12: "primary_full_vs_direct.mcnemar_exact_p",
    13: "primary_full_vs_direct.mcnemar_exact_p",
}


class _Cell:
    def __init__(self, group, label, source, n, wins, losses):
        self.group, self.label, self.source, self.n, self.wins, self.losses = group, label, source, n, wins, losses


# The 14 principal paired cells of the archive: source JSON (relative to the archive records) and the fields that hold
# the discordant counts.
PRINCIPAL_CELLS = [
    _Cell('controlled', 'ProofWriter / Qwen3-14B confirmation', 'evidence/qwen3_14b_confirmation/paired_statistics.json', 1000, 'candidate_only_correct', 'direct_only_correct'),
    _Cell('controlled', 'ProofWriter D5 / Qwen3-14B', 'evidence/logiclm_proofwriter_matched_v1/paired_statistics.json', 600, 'candidate_only_correct', 'direct_only_correct'),
    _Cell('controlled', 'PrOntoQA-OOD / Qwen compiler', 'evidence/archive_ood_v1/archive_success_gate.json', 800, 'metrics.paired.full_context_direct.candidate_only_correct', 'metrics.paired.full_context_direct.baseline_only_correct'),
    _Cell('diagnostic', 'ProofWriter / Qwen3-8B specific vs generic', 'evidence/qwen8_generic_feedback_dev/specific_vs_generic_paired.json', 1000, 'candidate_only_correct', 'direct_only_correct'),
    _Cell('diagnostic', 'ProofWriter / Qwen3-14B specific vs generic', 'results/PROOFWRITER_FEEDBACK_CROSS_SCALE_RESULT_V1.json', 1000, 'result.qwen3_14b_candidate_only_correct', 'result.qwen3_14b_generic_only_correct'),
    _Cell('diagnostic', 'ProofWriter / Llama-3.1-8B specific vs generic', 'results/PROOFWRITER_LLAMA31_FEEDBACK_INDEPENDENT_CONFIRMATION_RESULT_V1.json', 1000, 'llama31_8b_confirmation.candidate_only_correct', 'llama31_8b_confirmation.generic_only_correct'),
    _Cell('diagnostic', 'ProofWriter / Mistral-7B specific vs generic', 'results/PROOFWRITER_NONQWEN_FEEDBACK_CROSS_FAMILY_RESULT_V1.json', 1000, 'result.mistral7b.candidate_only_correct', 'result.mistral7b.generic_only_correct'),
    _Cell('SQL', 'Spider / Llama-3.1-8B', 'results/SPIDER_LLAMA31_GENERIC_FORWARD_CANONICAL_RESULT_V1.json', 2147, 'public_test_confirmation.candidate_only_correct', 'public_test_confirmation.initial_only_correct'),
    _Cell('SQL', 'Spider / Mistral-7B', 'results/SPIDER_GENERIC_FORWARD_CROSS_MODEL_RESULT_V1.json', 2147, 'mistral7b.candidate_only_correct', 'mistral7b.initial_only_correct'),
    _Cell('SQL', 'Spider / Qwen3-8B', 'results/SPIDER_GENERIC_FORWARD_CROSS_MODEL_RESULT_V1.json', 2147, 'qwen3_8b.candidate_only_correct', 'qwen3_8b.initial_only_correct'),
    _Cell('SQL', 'BIRD Mini-Dev / Llama-3.1-8B', 'results/BIRD_MINIDEV_GENERIC_FORWARD_RESULT_V1.json', 500, 'llama31_8b.candidate_only_correct', 'llama31_8b.initial_only_correct'),
    _Cell('SQL', 'BIRD Mini-Dev / Mistral-7B', 'results/BIRD_MINIDEV_GENERIC_FORWARD_RESULT_V1.json', 500, 'mistral7b.candidate_only_correct', 'mistral7b.initial_only_correct'),
    _Cell('OOD SQL', 'Dr.Spider / Llama-3.1-8B', 'drspider_oneuse/llama31_8b/analysis.json', 15269, 'primary_full_vs_direct.candidate_only_correct', 'primary_full_vs_direct.direct_only_correct'),
    _Cell('OOD SQL', 'Dr.Spider / Mistral-7B', 'drspider_oneuse/mistral7b/analysis.json', 15269, 'primary_full_vs_direct.candidate_only_correct', 'primary_full_vs_direct.direct_only_correct')
]


class _Archive:
    PAIRED_CELLS = PRINCIPAL_CELLS

    @staticmethod
    def load(relative):
        import json
        return json.loads((jp.ARCHIVE / relative).read_text(encoding="utf-8"))

    @staticmethod
    def at(value, dotted):
        for key in dotted.split("."):
            value = value[int(key)] if key.isdigit() else value[key]
        return value


def load_reproduce():
    return _Archive


def main():
    assert (PERM_B, PERM_SEED) == (20_000, 20261011)  # caption text below hard-codes these
    rep = load_reproduce()
    cells = rep.PAIRED_CELLS[:14]
    _, clusters, subset = safe_rows()
    rows = []
    for i, c in enumerate(cells):
        data = rep.load(c.source)
        p = float(rep.at(data, P_PATH[i]))
        w, l = int(rep.at(data, c.wins)), int(rep.at(data, c.losses))
        p_check = paired_stats.exact_mcnemar_p(w, l)
        assert abs(p - p_check) <= 1e-12 * max(1.0, p) or p == p_check, (c.label, p, p_check)
        row = {"family": "principal", "cell": f"{c.group}: {c.label}", "n": c.n, "cand_only": w,
               "comp_only": l, "raw_p": p, "raw_p_source": f"{c.source}::{P_PATH[i]}"}
        if c.label.startswith("Dr.Spider"):
            m = "llama31_8b" if "Llama" in c.label else "mistral7b"
            cand = correctness(DR / m / "cases" / "full.cases.jsonl")
            comp = correctness(DR / m / "cases" / "direct_full.cases.jsonl")
            assert len(cand) == 15_269
            st = paired_stats.paired(cand, comp, samples=1)  # only for discordant counts / p
            assert (st["candidate_only_correct"], st["baseline_only_correct"]) == (w, l)
            row.update(sign_flip_p(cand, comp, clusters))
        rows.append(row)
    for m in MODELS:
        an = load_json(DR / m / "analysis.json")["subset_comparisons"]
        cand = correctness(DR / m / "cases" / "subset_full.cases.jsonl")
        for comp_name in COMPARATORS:
            s = an[f"full_vs_{comp_name}"]
            comp = correctness(DR / m / "cases" / f"subset_{comp_name}.cases.jsonl")
            assert set(cand) == set(comp) == subset
            st = paired_stats.paired(cand, comp, samples=1)
            assert (st["candidate_only_correct"], st["baseline_only_correct"]) == \
                (s["candidate_only_correct"], s["direct_only_correct"])
            assert abs(st["mcnemar_exact_p"] - s["mcnemar_exact_p"]) <= 1e-15
            row = {"family": "ablation", "cell": f"Dr.Spider subset / {MODEL_LABEL[m]}: JIT-Repair vs {comp_name}",
                   "n": s["records"], "cand_only": s["candidate_only_correct"],
                   "comp_only": s["direct_only_correct"], "raw_p": s["mcnemar_exact_p"],
                   "raw_p_source": f"drspider_oneuse/{m}/analysis.json::subset_comparisons.full_vs_{comp_name}.mcnemar_exact_p"}
            row.update(sign_flip_p(cand, comp, clusters))
            rows.append(row)
    assert len(rows) == 26
    key = lambda r: r["cell"]
    g = paired_stats.holm({key(r): r["raw_p"] for r in rows})
    for fam in ("principal", "ablation"):
        h = paired_stats.holm({key(r): r["raw_p"] for r in rows if r["family"] == fam})
        for r in rows:
            if r["family"] == fam:
                r["holm_within_family"] = h[key(r)]
    # sensitivity: substitute the cluster permutation p for every Dr.Spider row, then Holm over 26
    sub = {key(r): (r["cluster_perm_p"] if "cluster_perm_p" in r else r["raw_p"]) for r in rows}
    gs = paired_stats.holm(sub)
    for r in rows:
        r["holm_global"] = g[key(r)]
        r["survives_global"] = g[key(r)] < 0.05
        r["holm_global_cluster_substituted"] = gs[key(r)]
        r["survives_global_cluster_substituted"] = gs[key(r)] < 0.05
    fields = ["family", "cell", "n", "cand_only", "comp_only", "raw_p", "holm_within_family", "holm_global",
              "survives_global", "cluster_perm_p", "perm_exceed", "perm_B", "perm_seed", "clusters",
              "nonzero_clusters", "holm_global_cluster_substituted", "survives_global_cluster_substituted",
              "raw_p_source"]
    write_csv(TABLES / "T4_global_holm.csv", rows, fields)
    L = [r"\begin{table}[t]\centering\scriptsize",
         r"\caption{Multiplicity control over the 14 principal paired cells and the 12 Dr.Spider ablation contrasts (26 tests). "
         r"Raw $p$: exact McNemar from the frozen JSON (ignores clustering for Dr.Spider). Holm adjustment within each family and jointly over all 26. "
         r"Cluster $p$: two-sided sign-flip permutation over original-question clusters (\texttt{q\_id\_spider\_dev}; 1{,}034 clusters full set, 780 subset), "
         r"$B=20{,}000$, seed 20261011; smallest attainable value $1/(B+1)\approx 5.0\times10^{-5}$.}",
         r"\label{tab:t4-global-holm}",
         r"\begin{tabular}{llrrrrrcr}\toprule",
         r"Family & Cell & $n$ & $N_{01}/N_{10}$ & Raw $p$ & Holm (family) & Holm (all 26) & $<0.05$ & Cluster $p$ \\ \midrule"]
    for r in rows:
        cp = fmt_p(r["cluster_perm_p"]) if "cluster_perm_p" in r else "--"
        cell = r["cell"].split(": ", 1)[1] if r["family"] == "principal" else r["cell"].replace("Dr.Spider subset / ", "")
        n_tex = f"{r['n']:,}".replace(",", "{,}")
        L.append(f"{r['family']} & {tex_escape(cell)} & {n_tex} & {r['cand_only']}/{r['comp_only']} & "
                 f"{fmt_p(r['raw_p'])} & {fmt_p(r['holm_within_family'])} & {fmt_p(r['holm_global'])} & "
                 f"{'yes' if r['survives_global'] else 'no'} & {cp} \\\\")
        if r is rows[13]:
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    (TABLES / "T4_global_holm.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['family']:9} {r['cell'][:62]:62} n={r['n']:>6} {r['cand_only']:>4}/{r['comp_only']:<4} "
              f"raw={r['raw_p']:.3g} hf={r['holm_within_family']:.3g} hg={r['holm_global']:.3g} "
              f"surv={r['survives_global']} cp={r.get('cluster_perm_p','')} hgc={r['holm_global_cluster_substituted']:.3g}")


if __name__ == "__main__":
    main()
