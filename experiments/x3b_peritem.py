"""Re-score the X3b/X3c gold-free selections (Dr.Spider subset) and DUMP per-item correctness for every method
(the first analysis kept only aggregates). Same evaluator + guard as x3b_drspider.analyze."""
import json, multiprocessing as mp, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3b_drspider import IN, OUT, TS, jl, _init, _score  # noqa


def main():
    safe = jl(IN / "safe_subset.jsonl"); by = {r["case_id"]: r for r in safe}; ids = [r["case_id"] for r in safe]
    pos = {c: i for i, c in enumerate(ids)}
    gold = {r["case_id"]: r["sql"] for r in jl(IN / "gold_subset.jsonl")}
    pool = mp.get_context("spawn").Pool(48, initializer=_init, initargs=(TS,))
    for model in ("llama31_8b", "mistral7b", "qwen3_8b"):
        sel = json.load(open(OUT / model / "selections_gold_free.json"))
        uniq = sorted({(k.split("|")[0], v[0]) for k, v in sel.items()})
        sc = pool.map(_score, [(pos[c], by[c], gold[c], s) for c, s in uniq], chunksize=8)
        score = {k: v[0] for k, v in zip(uniq, sc)}
        methods = sorted({k.split("|")[1] for k in sel})
        with open(OUT / model / "per_item.jsonl", "w") as h:
            for c in ids:
                h.write(json.dumps({"case_id": c, "q_id": by[c]["q_id_spider_dev"],
                                    **{m: score[(c, sel[f"{c}|{m}"][0])] for m in methods}}) + "\n")
        print(model, {m: round(sum(score[(c, sel[f'{c}|{m}'][0])] for c in ids) / len(ids) * 100, 2) for m in ("greedy", "jit", "vote@3", "gated_vote@3")}, flush=True)


if __name__ == "__main__":
    main()
