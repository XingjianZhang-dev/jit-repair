"""C1, Llama-3.1-8B step-by-step reasoning: scoring of the declared second attempt (Table 4 note, Supplement S4.5).

The step-by-step outputs still truncated at 4,096 tokens were regenerated once at 8,192 tokens
(llama_cot_second_attempt.py). Pre-registered scoring: an output that is still truncated counts as wrong. Worst case for
JIT-Repair: every still-truncated output counts as correct. Both are compared with JIT-Repair (specific revision, three
rounds) using the paired bootstrap of the C1 analysis. Writes tables/G1_llama_second_attempt.json under the records."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jit_repair import paths as jp  # noqa: E402
import paired_stats as st  # noqa: E402
import pw_tools as pt  # noqa: E402
from c1_analyze import LAB4, corr, jl  # noqa: E402


def main():
    d = jp.RECORDS / "results" / "pw" / "d5" / "llama31_8b"
    jit = corr(jl(d / "dsl_specific_merged_3.jsonl"))
    cot = {r["case_id"]: r for r in jl(d / "cot.jsonl")}
    second = {r["id"]: r["samples"][0] for r in jl(d / "second_attempt" / "gen_cot_8192.jsonl")}
    rule, worst, still = {}, {}, 0
    for case, row in cot.items():
        text, finish = (second[case]["text"], second[case]["finish_reason"]) if case in second else (row["raw_output"], row["finish_reason"])
        label = pt.extract_label(text)
        correct = int(label in LAB4 and label == row["gold"])
        truncated = finish == "length"
        still += truncated
        rule[case] = 0 if truncated else correct
        worst[case] = 1 if truncated else correct
    out = {"rule": st.paired(jit, rule, seed=20261011),
           "worst_case_truncated_correct": st.paired(jit, worst, seed=20261011),
           "truncated_after_second_attempt": still, "n": len(cot)}
    with open(jp.RECORDS / "tables" / "G1_llama_second_attempt.json", "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps({k: (v["difference"] if isinstance(v, dict) else v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
