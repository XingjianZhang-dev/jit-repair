"""X3: is JIT-Repair complementary to sampling? Apply JIT's gold-free trigger
(frozen checker + critic) and frozen generic repair + selector to the exec-vote@N winner.
Phase 'gen' (GPU, gold-free) writes combo records; phase 'score' (gold) scores and compares."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402
import argparse, json, multiprocessing as mp, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_c1_c2 import DB, SAFE, MAXTOK, MODELS, jl, wjl, _check  # noqa

R = jp.OUTBOX


def split_ids(bench, model, ids, split):
    if split == "discovery":
        return [c for i, c in enumerate(ids) if bench == "spider" and model == "llama31_8b" and i % 2 == 0]
    if split == "confirmation":
        if bench == "spider":
            return [c for i, c in enumerate(ids) if i % 2 == 1]
        return list(ids)
    raise ValueError(split)


def gen(model, cells, split, Ns):
    import sql_tools
    from gen_vllm import Engine
    from jit_repair import spider_feedback as sf, bird_minidev_feedback as bf
    from jit_repair.merge_spider_critic_retries import choose
    pool = mp.get_context("spawn").Pool(48)
    eng = Engine(MODELS[model], model, max_model_len=12288)
    for bench in cells:
        d = R / "c2" / bench / model
        safe = jl(SAFE[bench]); by = {str(r["case_id"]): r for r in safe}; ids = [str(r["case_id"]) for r in safe]
        prompts = {p["id"]: p for p in sql_tools.build_prompts(bench, safe, DB[bench])}
        checked = {r["case_id"]: r["samples"] for r in jl(d / "samples_checked.jsonl")}
        initial = {r["case_id"]: r for r in jl(d / "initial.jsonl")}
        sel_ids = split_ids(bench, model, ids, split)
        for N in Ns:
            out = R / "x3" / split / bench / model / f"combo_vote{N}.jsonl"
            if out.exists():
                continue
            winners = {}
            for c in sel_ids:
                g = {"sql": initial[c]["sql"], "execution_valid": initial[c]["execution_valid"]}
                ch, idx, calls = sql_tools.exec_vote(checked[c], N, g)
                tok = sum(s["output_tokens"] for s in checked[c][:N]) + (initial[c]["output_tokens"] if idx == "greedy" else 0)
                winners[c] = (ch["sql"], calls, tok, idx)
            chk = pool.map(_check, [(bench, by[c], winners[c][0]) for c in sel_ids], chunksize=8)
            wrec = {c: x for c, x in zip(sel_ids, chk)}
            flagged = [c for c in sel_ids if wrec[c]["needs_repair"]]
            rows = []
            for c in flagged:
                row = by[c]
                if bench == "spider":
                    schema = sf.schema_snapshot(sf.locate_database(DB[bench], str(row["db_id"])))[0]
                    user = sf.render_repair_user(str(row["question"]), schema, wrec[c]["sql"], sf.render_generic_feedback())
                else:
                    user = bf.render_repair_user(str(row["question"]), str(row["evidence"]), str(row["schema"]), wrec[c]["sql"], bf.render_generic_feedback())
                rows.append({"id": c, "system": prompts[c]["system"], "user": user})
            gr = eng.run(rows, str(out.with_suffix(".gen_repair.jsonl")), temperature=0.0, max_tokens=MAXTOK[bench], tag=f"x3/{bench}/{N}") if rows else {}
            rchk = pool.map(_check, [(bench, by[x["id"]], gr[x["id"]]["samples"][0]["text"]) for x in rows], chunksize=8)
            reps = {}
            for x, cc in zip(rows, rchk):
                s = gr[x["id"]]["samples"][0]
                reps[x["id"]] = {**cc, "output_tokens": s.get("output_tokens_frozen_convention", s["output_tokens"])}
            recs = []
            for c in sel_ids:
                prev = wrec[c]
                if c in reps:
                    sel, reason = choose(prev, reps[c])
                else:
                    sel, reason = prev, "unflagged_previous_immutable"
                recs.append({"case_id": c, "vote_sql": winners[c][0], "vote_calls": winners[c][1], "vote_tokens": winners[c][2],
                             "vote_idx": winners[c][3], "flagged": c in reps, "final_sql": sel["sql"],
                             "final_valid": sel["execution_valid"], "vote_valid": prev["execution_valid"],
                             "selection_reason": reason,
                             "combo_calls": winners[c][1] + int(c in reps),
                             "combo_tokens": winners[c][2] + (reps[c]["output_tokens"] if c in reps else 0)})
            wjl(out, recs)
            print("X3 gen", split, bench, model, N, "n", len(recs), "flagged", len(flagged), flush=True)
    pool.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--benches", default="spider,bird")
    ap.add_argument("--split", required=True); ap.add_argument("--Ns", default="3,5")
    a = ap.parse_args()
    gen(a.model, a.benches.split(","), a.split, [int(x) for x in a.Ns.split(",")])


if __name__ == "__main__":
    main()
