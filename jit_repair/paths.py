"""Directory layout shared by every script. Each root can be moved with an environment variable.

Generation (GPU machine), under ``JIT_WORK`` (default ``/workspace``):
    inbox/            prepared inputs without reference answers (DATA.md lists every file with its sha256)
    outbox/results/   generation outputs and per-question records, one directory per experiment
    data/             benchmarks and the official evaluators (scripts/download_data.sh)
    models/           model weights (scripts/download_models.sh)

Analysis (CPU), under ``JIT_RECORDS`` (default ``records/`` in this repository):
    results/          a copy of outbox/results
    tables/           tables written by the analysis scripts
    inputs/           the prepared Dr.Spider subset (inbox/drspider/safe_subset.jsonl)
    LEDGER.csv        the pre-registered predictions with their rules and verdicts (Supplement S5)
    archive/          records of the earlier runs on the second hardware stack (``JIT_ARCHIVE``)

Tables, figure data and figures produced by ``analysis/`` and ``figures/`` go to ``JIT_OUTPUTS`` (default ``outputs/``).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("JIT_WORK", "/workspace"))
INBOX = WORK / "inbox"
OUTBOX = WORK / "outbox" / "results"
DATA = WORK / "data"
MODELS = WORK / "models"
RECORDS = Path(os.environ.get("JIT_RECORDS", REPO / "records"))
ARCHIVE = Path(os.environ.get("JIT_ARCHIVE", RECORDS / "archive"))
OUTPUTS = Path(os.environ.get("JIT_OUTPUTS", REPO / "outputs"))


def outputs(sub: str) -> Path:
    """Return the output directory ``OUTPUTS / sub``, creating it if needed."""
    path = OUTPUTS / sub
    path.mkdir(parents=True, exist_ok=True)
    return path
