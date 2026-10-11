"""Synthetic known-answer tests for C2 outcome-deciding code (canonicalisation, voting,
filtering, evaluator wrappers). Must pass before any C2 data is scored."""
import os, sqlite3, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import sql_tools  # noqa


def make_db():
    d = tempfile.mkdtemp()
    p = Path(d) / "t.sqlite"
    c = sqlite3.connect(p)
    c.execute("create table a(x int, y text, z real)")
    c.executemany("insert into a values (?,?,?)", [(1, "p", 1.0), (2, "q", 2.5), (2, "q", 2.5), (3, "r", None)])
    c.commit(); c.close()
    return p


def test_canonical_keys():
    p = make_db()
    k = lambda s: sql_tools.result_key(p, s, 5.0)
    assert k("select x,y from a order by x") == k("select x,y from a order by x desc")      # row order ignored
    assert k("select x from a") != k("select distinct x from a")                              # multiset, not set
    assert k("select 1") == k("select 1.0")                                                   # 1 == 1.0
    assert k("select x,y from a")[1] != k("select y,x from a")[1]                             # column order kept
    assert k("select z from a where z is null")[0] == "ok"
    assert k("select nosuch from a") == ("error", None)
    assert k("select x from a where x > 99")[0] == "ok"                                       # empty result ok
    st, _ = sql_tools.result_key(p, "with recursive c(n) as (select 1 union all select n+1 from c) select count(*) from c", 0.5)
    assert st == "timeout"


def S(valid, key=None, sql=""):
    return {"execution_valid": valid, "key_status": "ok" if key else ("error" if not valid else "ok"),
            "result_key": key, "sql": sql}


def test_vote_and_filter():
    g = {"sql": "GREEDY"}
    a, b, c = S(True, "k1", "A"), S(True, "k2", "B"), S(True, "k1", "C")
    bad = S(False, None, "BAD")
    ch, idx, calls = sql_tools.exec_vote([a, b, c], 3, g); assert ch["sql"] == "A" and idx == 0 and calls == 3
    ch, idx, _ = sql_tools.exec_vote([b, a, c], 3, g); assert ch["sql"] == "A" and idx == 1      # k1 majority, earliest k1 member
    ch, idx, _ = sql_tools.exec_vote([a, b], 2, g); assert ch["sql"] == "A"                       # tie -> earliest group
    ch, idx, _ = sql_tools.exec_vote([b, a], 2, g); assert ch["sql"] == "B"
    ch, idx, _ = sql_tools.exec_vote([bad, b, a, c], 4, g); assert ch["sql"] == "A"               # invalid skipped
    ch, idx, calls = sql_tools.exec_vote([bad, bad], 2, g); assert ch["sql"] == "GREEDY" and idx == "greedy" and calls == 3
    ch, idx, _ = sql_tools.exec_vote([a, b, c], 1, g); assert ch["sql"] == "A"                    # prefix respected
    ch, idx, _ = sql_tools.exec_vote([b, a, c], 2, g); assert ch["sql"] == "B"                    # prefix N=2 -> tie -> B
    ch, idx, calls = sql_tools.exec_filter([bad, b, a], 3, g); assert ch["sql"] == "B" and idx == 1 and calls == 2
    ch, idx, calls = sql_tools.exec_filter([bad, bad, a], 2, g); assert ch["sql"] == "GREEDY" and calls == 3
    # valid by checker but key error (e.g. too many rows) must not vote
    weird = {"execution_valid": True, "key_status": "too_many_rows", "result_key": None, "sql": "W"}
    ch, idx, _ = sql_tools.exec_vote([weird, b], 2, g); assert ch["sql"] == "B"


def test_bird_scorer():
    p = make_db()
    assert sql_tools.score_bird(str(p), "select x from a", "select x from a order by x desc", True)[0] == 1
    assert sql_tools.score_bird(str(p), "select distinct x from a", "select x from a", True)[0] == 1   # set semantics (official)
    assert sql_tools.score_bird(str(p), "select y from a", "select x from a", True)[0] == 0
    assert sql_tools.score_bird(str(p), "select x from a", "select x from a", False)[0] == 0           # invalid -> 0
    assert sql_tools.score_bird(str(p), "select nosuch", "select x from a", True)[0] == 0


def test_guard():
    p = make_db()
    big = "select a1.x from " + ", ".join(f"a a{i}" for i in range(1, 10))   # 4^9 = 262,144 rows > 200,000
    assert sql_tools.guard(p, big, 5.0) == "too_many_rows"
    assert sql_tools.score_bird(str(p), big, "select x from a", True) == (0, "guard_too_many_rows")
    assert sql_tools.guard(p, "select x from a", 5.0) == "ok"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("PASS", name)
    print("ALL_SQL_TOOLS_TESTS_PASS")
