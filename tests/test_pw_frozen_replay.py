"""Replay check: the wrapper (pw_tools.validate_dsl / merge / strict_label) must reproduce the
frozen per-example prediction/error for archived outputs (CPU, no inference)."""
from jit_repair import paths as jp  # noqa: E402
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import pw_tools as pt  # noqa
E = jp.ARCHIVE / "evidence"

def jl(p): return [json.loads(l) for l in open(p)]

def test_replay(bench_path, out_dir, initial="dsl_initial.jsonl", rounds=("dsl_repair_round_1.jsonl",), direct="direct.jsonl"):
    bench = {r["case_id"]: r for r in jl(bench_path)}
    n = bad = 0
    for fname in (initial,) + tuple(rounds):
        for r in jl(out_dir / fname):
            pred, err, _ = pt.validate_dsl(bench[r["case_id"]], r["raw_output"])
            n += 1; bad += int(pred != r["prediction"] or err != r["error"])
    for r in jl(out_dir / direct):
        n += 1; bad += int(pt.strict_label(r["raw_output"]) != r["prediction"])
    return n, bad

if __name__ == "__main__":
    d5 = Path(sys.argv[1]); conf = Path(sys.argv[2])
    n, bad = test_replay(d5, E / "logiclm_proofwriter_matched_v1", rounds=("dsl_repair_round_1.jsonl", "dsl_repair_round_2.jsonl", "dsl_repair_round_3.jsonl"))
    print("D5 Q14 replay", n, "mismatches", bad); assert bad == 0
    n, bad = test_replay(conf, E / "qwen3_14b_confirmation", rounds=("dsl_repair_1.jsonl", "dsl_repair_2.jsonl", "dsl_repair_3.jsonl"))
    print("CONF Q14 replay", n, "mismatches", bad); assert bad == 0
    print("ALL_PW_REPLAY_PASS")
