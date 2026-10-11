"""T3: like-for-like generation cost of JIT-Repair vs Self-Refine on the SAME 1,556-item Dr.Spider subset.

JIT-Repair (arm subset_full, build_drspider_oneuse_arms.py): initial generation for every subset item
+ one generic repair call for each subset item whose initial output the gold-free critic flagged
(needs_repair). The generic_repair.jsonl file also holds repairs for UNflagged subset items (used only by
subset_always_generic); those are NOT part of the JIT-Repair arm and are excluded here.
Self-Refine (arm subset_self_refine): initial generation + feedback call + iterate call per subset item.
Cost fields per raw row: generation_latency_seconds (= model.generate wall time of a batch of 4 / batch
size), input_tokens, output_tokens (identical to analyze_drspider_oneuse._generation_totals)."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from sql_archive_common import (DR, MODELS, MODEL_LABEL, TABLES, keyed, load_json, safe_rows, correctness,
                    write_csv, tex_escape)

METRICS = ("model_calls", "output_tokens", "input_tokens", "generation_seconds")


def totals(rows):
    return {"model_calls": len(rows),
            "output_tokens": sum(int(r["output_tokens"]) for r in rows),
            "input_tokens": sum(int(r["input_tokens"]) for r in rows),
            "generation_seconds": sum(float(r["generation_latency_seconds"]) for r in rows)}


def add(*ts):
    return {k: sum(t[k] for t in ts) for k in METRICS}


def main():
    _, clusters, subset = safe_rows()
    out, checks = [], []
    for m in MODELS:
        raw = DR / m / "raw"
        init = keyed(raw / "initial.jsonl")
        gen = keyed(raw / "generic_repair.jsonl")
        fb = keyed(raw / "self_refine_feedback.jsonl")
        it = keyed(raw / "self_refine_iterate.jsonl")
        assert set(fb) == subset and set(it) == subset
        flagged_sub = sorted(c for c in subset if bool(init[c]["needs_repair"]))
        assert all(c in gen for c in flagged_sub)
        # cross-check against the frozen subset_full arm: every attempt-2 row is a generic repair of a
        # flagged subset item, and the selector saw a repair for every flagged subset item
        arm = keyed(DR / m / "arms" / "subset_full.jsonl")
        assert set(arm) == subset
        acc_rows = [c for c, r in arm.items() if int(r["attempt"]) == 2]
        assert set(acc_rows) <= set(flagged_sub)
        unflag_reason = {arm[c]["selection_reason"] for c in subset if c not in set(flagged_sub)}
        assert unflag_reason == {"unflagged_initial_immutable"}, unflag_reason
        audit = load_json(DR / m / "arms_audit.json")
        assert audit["critic_flagged_subset"] == len(flagged_sub)

        t_init = totals([init[c] for c in sorted(subset)])
        t_rep = totals([gen[c] for c in flagged_sub])
        t_fb = totals([fb[c] for c in sorted(subset)])
        t_it = totals([it[c] for c in sorted(subset)])
        jit = add(t_init, t_rep)
        sr = add(t_init, t_fb, t_it)
        sr_marg = add(t_fb, t_it)
        # reproduce the frozen analysis.json efficiency block
        eff = load_json(DR / m / "analysis.json")["efficiency"]
        for key, ours in (("subset_initial", t_init), ("subset_self_refine_end_to_end", sr)):
            ref = eff[key]
            ok = (ref["records"] == ours["model_calls"] and ref["output_tokens"] == ours["output_tokens"]
                  and ref["input_tokens"] == ours["input_tokens"]
                  and abs(ref["generation_seconds"] - ours["generation_seconds"]) < 1e-6)
            checks.append(f"{m} {key} matches analysis.json: {ok}")
            assert ok
        paper_full_ratio = eff["full_to_direct_generation_seconds_ratio"]
        paper_sr_ratio = eff["self_refine_to_subset_direct_generation_seconds_ratio"]

        acc = {}
        for name in ("subset_direct", "subset_full", "subset_self_refine"):
            c = correctness(DR / m / "cases" / f"{name}.cases.jsonl")
            assert set(c) == subset
            acc[name] = (sum(c.values()), len(c))
        n = len(subset)
        label = MODEL_LABEL[m]
        arms = (
            ("Direct (initial only) = subset_direct", "value", t_init, acc["subset_direct"]),
            ("JIT-Repair = subset_full (initial + repairs of flagged)", "value", jit, acc["subset_full"]),
            ("Self-Refine = subset_self_refine (initial + feedback + iterate)", "value", sr, acc["subset_self_refine"]),
            ("JIT-Repair repair stage only (flagged repairs)", "value", t_rep, None),
            ("Self-Refine refinement stage only (feedback + iterate)", "value", sr_marg, None),
        )
        for name, kind, t, a in arms:
            row = {"model": label, "row": name, "kind": kind, "items": n,
                   "flagged_items": len(flagged_sub)}
            for k in METRICS:
                row[f"{k}_total"] = t[k] if k != "generation_seconds" else round(t[k], 3)
                row[f"{k}_per_item"] = round(t[k] / n, 4)
            if a:
                row["n_correct"], row["n"] = a
                row["accuracy"] = round(a[0] / a[1], 6)
            out.append(row)
        for name, num, den in (("ratio Self-Refine / JIT-Repair", sr, jit),
                               ("ratio JIT-Repair / Direct", jit, t_init),
                               ("ratio Self-Refine / Direct (paper's 3.199x/3.228x quantity)", sr, t_init),
                               ("ratio SR refinement stage / JIT repair stage", sr_marg, t_rep)):
            row = {"model": label, "row": name, "kind": "ratio", "items": n, "flagged_items": len(flagged_sub)}
            for k in METRICS:
                row[f"{k}_total"] = round(num[k] / den[k], 4)
            out.append(row)
        checks.append(f"{m}: paper full-set JIT/direct gen-seconds ratio {paper_full_ratio:.4f} (n=15,269); "
                      f"paper SR/subset-direct {paper_sr_ratio:.4f} (n=1,556); like-for-like SR/JIT "
                      f"{sr['generation_seconds']/jit['generation_seconds']:.4f}, JIT/direct subset "
                      f"{jit['generation_seconds']/t_init['generation_seconds']:.4f}")
    fields = ["model", "row", "kind", "items", "flagged_items"] + \
        [f"{k}_{s}" for k in METRICS for s in ("total", "per_item")] + ["n_correct", "n", "accuracy"]
    write_csv(TABLES / "T3_like_for_like_cost.csv", out, fields)

    # LaTeX
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Like-for-like generation cost on the same Dr.Spider ablation subset ($n=1{,}556$ perturbations, 780 original-question clusters). "
         r"Seconds are summed \texttt{generation\_latency\_seconds} (wall time of each batch of 4 divided by 4; excludes model load and evaluation). "
         r"JIT-Repair = initial generation for every item plus one generic repair call per critic-flagged item.}",
         r"\label{tab:t3-like-for-like-cost}",
         r"\begin{tabular}{llrrrrr}\toprule",
         r"Model & Arm & Calls/item & Out tok/item & In tok/item & Sec/item & Acc. (\%) \\ \midrule"]
    for r in out:
        if r["kind"] == "value" and "stage only" not in r["row"]:
            arm = r["row"].split(" =")[0]
            accs = f"{100*r['accuracy']:.2f} ({r['n_correct']}/{r['n']})"
            L.append(f"{tex_escape(r['model'])} & {tex_escape(arm)} & {r['model_calls_per_item']:.3f} & "
                     f"{r['output_tokens_per_item']:.1f} & {r['input_tokens_per_item']:.1f} & "
                     f"{r['generation_seconds_per_item']:.3f} & {accs} \\\\")
        elif r["kind"] == "ratio" and r["row"].startswith(("ratio Self-Refine / JIT", "ratio JIT-Repair / Direct")):
            nm = "SR / JIT" if "Self-Refine / JIT" in r["row"] else "JIT / Direct"
            L.append(f"{tex_escape(r['model'])} & \\textit{{{nm} ratio}} & {r['model_calls_total']:.2f}$\\times$ & "
                     f"{r['output_tokens_total']:.2f}$\\times$ & {r['input_tokens_total']:.2f}$\\times$ & "
                     f"{r['generation_seconds_total']:.2f}$\\times$ & \\\\")
            if nm == "JIT / Direct" and r["model"] == "Llama-3.1-8B":
                L.append(r"\midrule")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    (TABLES / "T3_like_for_like_cost.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(checks))
    for r in out:
        print({k: v for k, v in r.items() if v != ""})


if __name__ == "__main__":
    main()
