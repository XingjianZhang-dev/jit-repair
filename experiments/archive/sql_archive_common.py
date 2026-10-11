"""Shared read-only helpers for the archive analyses (T3, T4, T6).

Reads the archive records only; writes only to the records' tables/ directory."""
from __future__ import annotations
from jit_repair import paths as jp  # noqa: E402

import csv
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REC = jp.RECORDS
TABLES = REC / "tables"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_stats  # noqa: E402

import numpy as np  # noqa: E402

EV = jp.ARCHIVE / "evidence"
DR = jp.ARCHIVE / "drspider_oneuse"
SAFE = EV / "drspider_oneuse_assets_v1" / "safe.jsonl"
MODELS = ("llama31_8b", "mistral7b")
MODEL_LABEL = {"llama31_8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B", "qwen3_8b": "Qwen3-8B"}
COMPARATORS = ("direct", "self_refine", "always_generic", "execution_only", "no_selector", "random_gate")
PERM_SEED = 20261011
PERM_B = 20_000


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def keyed(path: Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    out = {str(r["case_id"]): r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"duplicate case_id in {path}")
    return out


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_rows():
    rows = load_jsonl(SAFE)
    clusters = {str(r["case_id"]): int(r["q_id_spider_dev"]) for r in rows}
    subset = {str(r["case_id"]) for r in rows if bool(r["in_ablation_subset"])}
    assert len(rows) == 15_269 and len(subset) == 1_556, (len(rows), len(subset))
    return rows, clusters, subset


def correctness(cases_path: Path) -> dict[str, int]:
    return {cid: int(bool(r["correct"])) for cid, r in keyed(cases_path).items()}


def sign_flip_p(cand: dict, comp: dict, clusters: dict, seed: int = PERM_SEED, B: int = PERM_B,
                chunk: int = 1_000) -> dict:
    """Two-sided cluster sign-flip permutation test on per-cluster (cand - comp) correct-count sums.
    p = (1 + #{|T*| >= |T_obs|}) / (1 + B); rng = numpy default_rng(seed), fresh per call."""
    if set(cand) != set(comp):
        raise ValueError("arms differ in ids")
    ids = sorted(cand)
    cl = sorted({clusters[i] for i in ids})
    pos = {c: j for j, c in enumerate(cl)}
    d = np.zeros(len(cl), dtype=np.int64)
    for i in ids:
        d[pos[clusters[i]]] += int(cand[i]) - int(comp[i])
    t_obs = int(d.sum())
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < B:
        m = min(chunk, B - done)
        signs = rng.integers(0, 2, size=(m, len(cl)), dtype=np.int64) * 2 - 1
        t = signs @ d
        exceed += int(np.sum(np.abs(t) >= abs(t_obs)))
        done += m
    return {"cluster_perm_p": (1 + exceed) / (1 + B), "perm_exceed": exceed, "perm_B": B,
            "perm_seed": seed, "clusters": len(cl), "nonzero_clusters": int(np.sum(d != 0)),
            "T_obs": t_obs}


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p >= 0.1:
        return f"{p:.3f}"
    if p >= 0.001:
        return f"{p:.3g}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"
