"""X7 confirmation set: BIRD dev questions NOT in Mini-Dev (not used in any earlier run).
Fields and schema come from the official mini_dev repo's dev_prompt.jsonl (same source format as the frozen
Mini-Dev safe.jsonl). Identity check: for the 500 Mini-Dev items, dev_prompt.jsonl's schema must equal the frozen
safe.jsonl schema. Exclusion (gold-free): question_id in Mini-Dev, or normalised question text identical / token-
Jaccard >= 0.9 to any Mini-Dev question. Gold self-check: each gold SQL executed once (30 s); failures are fixed
incorrect (as in the archive). Writes safe.jsonl (no gold) and gold.jsonl separately, with sha256."""
from jit_repair import paths as jp  # noqa: E402
import hashlib, json, re, sqlite3, time
from pathlib import Path
MD = (jp.DATA / "dl/mini_dev/finetuning/inference")
DBR = (jp.DATA / "bird/dev_20240627/dev_databases")
OUT = (jp.OUTBOX / "x7/birddev_assets"); OUT.mkdir(parents=True, exist_ok=True)
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
dev = jl(MD / "dev_prompt.jsonl"); mini = jl(MD / "mini_dev_prompt.jsonl")
frozen = jl(str(jp.INBOX / "frozen/bird_assets/safe.jsonl"))
fz_by_q = {int(r["question_id"]): r for r in frozen}
dev_by_q = {int(r["question_id"]): r for r in dev}
# identity check on Mini-Dev items
same = sum(dev_by_q[q]["schema"] == fz_by_q[q]["schema"] for q in fz_by_q if q in dev_by_q)
present = sum(q in dev_by_q for q in fz_by_q)
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
mini_q = [set(norm(r["question"])) for r in mini]; mini_text = {" ".join(norm(r["question"])) for r in mini}
mini_ids = {int(r["question_id"]) for r in mini}
keep, excl = [], {"id": 0, "text": 0, "jaccard": 0}
for r in dev:
    q = int(r["question_id"])
    if q in mini_ids: excl["id"] += 1; continue
    t = norm(r["question"])
    if " ".join(t) in mini_text: excl["text"] += 1; continue
    st = set(t)
    if any(len(st & m) / max(1, len(st | m)) >= 0.9 for m in mini_q): excl["jaccard"] += 1; continue
    keep.append(r)
def gold_ok(db, sql):
    con = sqlite3.connect((DBR / db / f"{db}.sqlite").resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    con.text_factory = lambda b: b.decode(errors="replace")
    dl = time.monotonic() + 30.0; con.set_progress_handler(lambda: int(time.monotonic() > dl), 10_000)
    try:
        con.execute(sql).fetchall(); return True
    except Exception:
        return False
    finally:
        con.close()
safe, gold, bad = [], [], []
for i, r in enumerate(keep):
    cid = f"bird-dev-{int(r['question_id'])}"
    safe.append({"case_id": cid, "source_index": i, "question_id": int(r["question_id"]), "db_id": r["db_id"],
                 "question": r["question"], "evidence": r["evidence"], "schema": r["schema"], "difficulty": r["difficulty"]})
    ok = gold_ok(r["db_id"], r["SQL"])
    gold.append({"case_id": cid, "db_id": r["db_id"], "sql": r["SQL"], "gold_executes": ok})
    if not ok: bad.append(cid)
def w(p, rows):
    with open(p, "w", encoding="utf-8") as h:
        for x in rows: h.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
audit = {"dev_prompt_records": len(dev), "mini_dev_records": len(mini), "minidev_in_dev_prompt": present,
         "minidev_schema_identical_to_frozen": same, "excluded": excl, "kept": len(safe),
         "gold_failures_fixed_incorrect": bad, "by_difficulty": {d: sum(s["difficulty"] == d for s in safe) for d in ("simple", "moderate", "challenging")},
         "by_db": {}, "safe_sha256": w(OUT / "safe.jsonl", safe), "gold_sha256": w(OUT / "gold.jsonl", gold)}
for s in safe: audit["by_db"][s["db_id"]] = audit["by_db"].get(s["db_id"], 0) + 1
json.dump(audit, open(OUT / "audit.json", "w"), indent=1)
print(json.dumps({k: v for k, v in audit.items() if k != "by_db"}, indent=1))
