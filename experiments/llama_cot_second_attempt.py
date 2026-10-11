"""C1 Llama declared second attempt: the 42 items still truncated at 4,096, greedy, max_new_tokens 8,192."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_c1_c2 import OUT, PW, MODELS, jl  # noqa
import pw_tools as pt  # noqa


def main():
    from gen_vllm import Engine
    d = OUT / "pw" / "d5" / "llama31_8b"
    cot = jl(d / "cot.jsonl"); trunc = [r["case_id"] for r in cot if r["finish_reason"] == "length"]
    bench = {r["case_id"]: r for r in jl(PW["d5"])}
    rows = [pt.cot_row(bench[c]) for c in trunc]
    eng = Engine(MODELS["llama31_8b"], "llama31_8b", max_model_len=10240)
    g = eng.run(rows, str(d / "second_attempt" / "gen_cot_8192.jsonl"), temperature=0.0, max_tokens=8192, tag="llama/cot-8192")
    print("items", len(rows), "still truncated", sum(g[c]["samples"][0]["finish_reason"] == "length" for c in trunc), flush=True)


if __name__ == "__main__":
    main()
