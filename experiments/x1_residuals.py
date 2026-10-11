"""X1: residual-error visibility after JIT-Repair (final records), archive cells and new-stack cells."""
from jit_repair import paths as jp  # noqa: E402
import json, os, sys
from pathlib import Path
E = jp.ARCHIVE / "evidence"; DR = jp.ARCHIVE / "drspider_oneuse"
REC = jp.RECORDS
def jl(p): return [json.loads(l) for l in open(p)]

def classify(final, correct, initial):
    cats = {"invisible": 0, "visible_critic_flag": 0, "visible_invalid": 0, "invisible_never_flagged": 0}
    resid = 0
    for c, r in final.items():
        if correct[c]: continue
        resid += 1
        if not r["execution_valid"]: cats["visible_invalid"] += 1
        elif r["critic_issues"]: cats["visible_critic_flag"] += 1
        else:
            cats["invisible"] += 1
            if not initial[c]["needs_repair"]: cats["invisible_never_flagged"] += 1
    return resid, cats

def cell_frozen_sb(final_p, ev_p, init_p):
    fin = {r["case_id"]: r for r in jl(final_p)}; ini = {r["case_id"]: r for r in jl(init_p)}
    ev = {r["case_id"]: bool(r["correct"]) for r in jl(ev_p)}
    return len(fin), *classify(fin, ev, ini)

cells = {}
SPL = E / "spider_generic_forward_canonical_v2_full"
cells["DISCOVERY spider llama31 (frozen)"] = cell_frozen_sb(SPL / "candidate/final.jsonl", SPL / "evaluation/candidate.cases.jsonl", SPL / "initial/initial.jsonl")
for m in ("llama31_8b", "mistral7b"):
    B = E / "bird_minidev_generic_forward_two_model_v1_full/results" / m
    cells[f"CONFIRM bird {m} (frozen)"] = cell_frozen_sb(B / "candidate/final.jsonl", B / "evaluation/candidate.cases.jsonl", B / "initial/initial.jsonl")
    fin = {r["case_id"]: r for r in jl(DR / m / "arms/full.jsonl")}
    ini = {r["case_id"]: r for r in jl(DR / m / "arms/direct_full.jsonl")}
    ev = {r["case_id"]: bool(r["correct"]) for r in jl(DR / m / "cases/full.cases.jsonl")}
    cells[f"CONFIRM drspider {m} (frozen)"] = (len(fin), *classify(fin, ev, ini))
for m in ("mistral7b", "qwen3_8b"):
    S = E / "spider_generic_forward_cross_model_public_test_v1_full" / m
    cells[f"DESCRIPTIVE spider {m} (frozen, same items)"] = cell_frozen_sb(S / "candidate/final.jsonl", S / "evaluation/candidate.cases.jsonl", S / "initial/initial.jsonl")
out = {}
for k, (n, resid, cats) in cells.items():
    share = cats["invisible"] / resid if resid else None
    evaluable = resid >= 50
    out[k] = {"n": n, "residual_errors": resid, **cats, "invisible_share": share, "evaluable": evaluable,
              "pass_gt_50pct": (share > 0.5) if (evaluable and k.startswith("CONFIRM")) else None}
    print(f"{k:45s} n={n} resid={resid} ({resid/n*100:.1f}%) invisible={cats['invisible']} ({share*100:.1f}%) never_flagged={cats['invisible_never_flagged']} critic={cats['visible_critic_flag']} invalid={cats['visible_invalid']}")
conf = [v for k, v in out.items() if k.startswith("CONFIRM")]
ev = [v for v in conf if v["evaluable"]]
verdict = "CONFIRMED" if len(ev) >= 3 and all(v["pass_gt_50pct"] for v in ev) else ("NOT CONFIRMED" if len(ev) >= 3 else "NOT EVALUABLE")
out["verdict"] = verdict; print("X1 verdict:", verdict)
json.dump(out, open(REC / "tables/X1_residual_errors.json", "w"), indent=1)
