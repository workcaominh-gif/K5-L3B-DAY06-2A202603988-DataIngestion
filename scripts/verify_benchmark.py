"""Check that every benchmark evidence quote occurs on its gold page (after pipeline normalization).

Run from repo root after `cd starter && python -m src.run inspect`:  python scripts/verify_benchmark.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from src.evaluation import norm  # noqa: E402

pages = {}
for line in (ROOT / "starter/data/processed/pages.jsonl").read_text(encoding="utf-8").splitlines():
    p = json.loads(line)
    pages[(p["document_id"], p["page"])] = p
ok = 0
qs = [json.loads(l) for l in (ROOT / "benchmark/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
for q in qs:
    golds = [(q["gold_source_id"], q["gold_page"], q["evidence_quote"])] + \
            [(a["source_id"], pg, a["evidence"]) for a in q.get("alternatives", []) for pg in a["pages"]]
    good = all(norm(ev) in norm(pages[(d, pg)]["text"]) for d, pg, ev in golds)
    fil = all(pages[(q["gold_source_id"], q["gold_page"])]["metadata"].get(k) == v for k, v in (q.get("filter") or {}).items())
    ok += good and fil
    if not (good and fil):
        print("FAIL", q["id"], "evidence" if not good else "filter")
print(f"{ok}/{len(qs)} questions verified")
