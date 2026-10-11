"""SQL helpers: frozen prompt rendering, frozen gold-free checker,
execution-result canonicalisation, exec-filter / exec-vote selection, and the
paper's evaluators (Spider: test-suite exec_eval on the canonical DB with
plug_value=False, keep_distinct=True; BIRD: set equality under a 30 s budget).
Gold is used ONLY by score_* functions."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

from jit_repair import spider_feedback as sf
from jit_repair import bird_minidev_feedback as bf
from jit_repair.spider_critic_feedback import critic_score, lint_question_sql

SPIDER_TIMEOUT = 10.0   # frozen checker timeout (Spider protocol.json)
BIRD_TIMEOUT = 30.0     # frozen checker/evaluator timeout (BIRD protocol.json)


# ---------------------------------------------------------------- prompts
def build_prompts(bench: str, safe_rows, db_dir: Path):
    rows, cache = [], {}
    for r in safe_rows:
        if bench == "spider":
            db = str(r["db_id"])
            if db not in cache:
                cache[db] = sf.schema_snapshot(sf.locate_database(db_dir, db))
            user = sf.render_initial_user(str(r["question"]), cache[db][0])
            system = sf.SQL_SYSTEM_PROMPT
        else:
            user = bf.render_initial_user(str(r["question"]), str(r["evidence"]), str(r["schema"]))
            system = bf.BIRD_SQL_SYSTEM_PROMPT
        rows.append({"id": str(r["case_id"]), "system": system, "user": user, "db_id": str(r["db_id"])})
    return rows


# ---------------------------------------------------------------- frozen checker
def check(bench: str, safe_row, raw: str, db_dir: Path):
    sql = sf.extract_sql(raw)
    loc = sf.locate_database if bench == "spider" else bf.locate_database
    timeout = SPIDER_TIMEOUT if bench == "spider" else BIRD_TIMEOUT
    ex = sf.execute_readonly(loc(db_dir, str(safe_row["db_id"])), sql, timeout)
    if bench == "spider":
        issues = lint_question_sql(str(safe_row["question"]), sql) if ex.valid else []
    else:
        issues = lint_question_sql(bf.critic_text(str(safe_row["question"]), str(safe_row["evidence"])), sql) if ex.valid else []
    return {"sql": sql, "execution_valid": ex.valid, "error": ex.error, "error_class": ex.error_class,
            "critic_issues": issues, "critic_score": critic_score(issues),
            "needs_repair": (not ex.valid) or bool(issues), "rows_observed_capped": ex.rows_observed}


# ---------------------------------------------------------------- canonicalisation (gold-free)
def _norm(v):
    if v is None:
        return ("n",)
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int):
        return ("num", repr(float(v)))
    if isinstance(v, float):
        if v != v:
            return ("nan",)
        r = round(v, 6)
        return ("num", repr(float(r)))
    if isinstance(v, bytes):
        return ("b", v.hex())
    return ("s", str(v))


def canonical_key(rows) -> str:
    """Multiset of rows (row order ignored, column order kept), numeric values unified
    (1 == 1.0, floats rounded to 6 decimals). Returns a sha256 hex digest."""
    normed = sorted(json.dumps([_norm(v) for v in row]) for row in rows)
    return hashlib.sha256(("\n".join(normed)).encode()).hexdigest()


def result_key(db_path: Path, sql: str, timeout: float, max_rows: int = 200000):
    """Execute read-only with a wall-clock budget; returns (status, key).
    status in {ok, timeout, error, too_many_rows}."""
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    con.text_factory = lambda raw: raw.decode(errors="replace")
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return "too_many_rows", None
        return "ok", canonical_key(rows)
    except sqlite3.OperationalError as e:
        return ("timeout" if "interrupted" in str(e) else "error"), None
    except Exception:
        return "error", None
    finally:
        con.close()


# ---------------------------------------------------------------- selection (gold-free)
def exec_filter(samples, N: int, greedy):
    """samples: list of dicts with 'execution_valid' (frozen checker) in sample order.
    Returns (chosen, index or 'greedy', calls_lazy) -- first executable among the first N,
    else the greedy output. calls_lazy = samples consumed sequentially (+1 greedy if fallback)."""
    for i, s in enumerate(samples[:N]):
        if s["execution_valid"]:
            return s, i, i + 1
    return greedy, "greedy", N + 1


def exec_vote(samples, N: int, greedy):
    """Majority over canonicalised execution results of executable samples among the first N
    (executable = frozen checker valid AND result_key status ok). Ties -> the group whose
    earliest member is earliest; returns that earliest member. No executable sample -> greedy."""
    groups, first = {}, {}
    for i, s in enumerate(samples[:N]):
        if s["execution_valid"] and s.get("key_status") == "ok":
            k = s["result_key"]
            groups[k] = groups.get(k, 0) + 1
            first.setdefault(k, i)
    if not groups:
        return greedy, "greedy", N + 1
    best = max(groups, key=lambda k: (groups[k], -first[k]))
    return samples[first[best]], first[best], N


# ---------------------------------------------------------------- paper evaluators (gold used here only)
def guard(db_path, sql: str, timeout: float, max_rows: int = 200000):
    """Evaluator-timeout guard: the official evaluators cannot interrupt a pathological prediction (e.g. a
    Cartesian product with billions of rows), which hangs/explodes memory; the evaluator's own semantics for a
    prediction that does not finish is 'incorrect'. Pre-execute the prediction read-only with a wall-clock budget
    and a row cap; a timeout or > max_rows result is scored incorrect without calling the evaluator."""
    st, _ = result_key(Path(db_path), sql, timeout, max_rows)
    return st


def score_spider(test_suite_repo: str, db_root: str, db_id: str, sql: str, gold_sql: str, execution_valid: bool):
    if not execution_valid:
        return 0, None
    g = guard(Path(db_root) / db_id / f"{db_id}.sqlite", sql, 10.0)
    if g in ("timeout", "too_many_rows"):
        return 0, f"guard_{g}"
    if test_suite_repo not in sys.path:
        sys.path.insert(0, test_suite_repo)
    try:
        from exec_eval import eval_exec_match
        db = str(Path(db_root) / db_id / f"{db_id}.sqlite")
        return int(eval_exec_match(db=db, p_str=sql, g_str=gold_sql, plug_value=False, keep_distinct=True,
                                   progress_bar_for_each_datapoint=False)), None
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def score_bird(db_path: str, sql: str, gold_sql: str, execution_valid: bool, timeout: float = 30.0):
    if not execution_valid:
        return 0, None
    g = guard(db_path, sql, timeout)
    if g in ("timeout", "too_many_rows"):
        return 0, f"guard_{g}"
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    con.text_factory = lambda raw: raw.decode(errors="replace")
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        p = con.execute(sql).fetchall()
        g = con.execute(gold_sql).fetchall()
        return int(set(p) == set(g)), None
    except sqlite3.Error as e:
        return 0, f"{type(e).__name__}: {e}"
    finally:
        con.set_progress_handler(None, 0)
        con.close()
