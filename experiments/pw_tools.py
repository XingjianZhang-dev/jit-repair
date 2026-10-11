"""ProofWriter helpers. All prompts, the DSL validator/executor and the
revision prompt come from jit_repair.run_model.
New here: the pre-registered C1 answer extractor for free-form reasoning outputs."""
from __future__ import annotations

import re

from jit_repair import run_model as rm

LABELS = ("true", "false", "both", "unknown")
# last 'FINAL: <label>' that is not an echo of the option list 'true|false|both|unknown'
_FINAL = re.compile(r"FINAL\s*:\s*[\*\s`\"'\[\(]*(true|false|both|unknown)\b(?![\s\*`\"'\]\)]*[|/])", re.I)


def strip_think(text: str):
    """Remove Qwen3 reasoning. Returns None if a reasoning block was opened but never closed
    (truncated thinking => no answer)."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None
    return text


def extract_label(text: str):
    """C1 primary extractor: last non-echo 'FINAL: label' after removing reasoning blocks."""
    t = strip_think(text)
    if t is None:
        return None
    m = _FINAL.findall(t)
    return m[-1].lower() if m else None


def strict_label(text: str):
    """Paper's frozen direct-arm parser (last non-empty line must be exactly 'FINAL: label'),
    applied after reasoning removal. Reported as a sensitivity only."""
    t = strip_think(text)
    return None if t is None else rm.parse_label(t)


# ---------------------------------------------------------------- frozen prompts
def direct_row(rec):
    return {"id": rec["case_id"], "system": rm.DIRECT_SYSTEM, "user": rm.render_user(rec)}


def cot_row(rec):
    return {"id": rec["case_id"], "system": rm.COT_SYSTEM, "user": rm.render_user(rec)}


def dsl_row(rec):
    return {"id": rec["case_id"], "system": rm.DSL_SYSTEM, "user": rm.render_dsl_user(rec)}


def repair_row(rec, previous, feedback):
    return {"id": rec["case_id"], "system": rm.DSL_SYSTEM,
            "user": rm.render_repair_content(rm.render_dsl_user(rec), previous, feedback)}


def validate_dsl(rec, text):
    """Exactly the frozen run_model compile_dsl branch."""
    error, metadata = None, {}
    try:
        ef, er = rm.expected_theory_counts(str(rec["theory"]))
        prediction, metadata = rm.execute_dsl(text, ef, er, str(rec["theory"]), str(rec["query"]))
    except Exception as exception:  # noqa: BLE001 -- frozen behaviour
        prediction = None
        error = f"{type(exception).__name__}: {exception}"
    return prediction, error, metadata


def merge(previous: dict, repairs: dict) -> dict:
    """Exactly the frozen merge_retries.py semantics."""
    out = {}
    for cid, first in previous.items():
        retry = repairs.get(cid)
        if retry is None:
            first = dict(first); first.setdefault("attempts", 1); out[cid] = first
        else:
            retry = dict(retry)
            retry["attempts"] = int(first.get("attempts", 1)) + 1
            retry["total_input_tokens"] = int(first.get("total_input_tokens", first["input_tokens"])) + int(retry["input_tokens"])
            retry["total_output_tokens"] = int(first.get("total_output_tokens", first["output_tokens"])) + int(retry["output_tokens"])
            retry["first_attempt_error"] = first.get("error")
            out[cid] = retry
    return out
