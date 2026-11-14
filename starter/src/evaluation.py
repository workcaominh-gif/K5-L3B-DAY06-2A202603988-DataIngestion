"""Automatic retrieval metrics (Hit@1, Evidence Hit@3) with full per-query traces.

A retrieved chunk is a HIT when:
  1. its document is a gold document, and
  2. its page range [page, page_end] contains a gold page, and
  3. it contains the FULL gold evidence quote (after whitespace / hyphenation normalization).
Any documented alternative gold (question["alternatives"]) is accepted too.
Coherence and grounding are NOT inferred here: they are labelled by hand (see results/manual_*.csv).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace("­", "")
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)          # hyphenation across lines
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _golds(q: dict[str, Any]) -> list[dict[str, Any]]:
    main = {"source_id": q["gold_source_id"], "pages": q["gold_page"] if isinstance(q["gold_page"], list) else [q["gold_page"]],
            "evidence": q["evidence_quote"]}
    return [main] + list(q.get("alternatives", []))


def is_hit(q: dict[str, Any], hit: dict[str, Any]) -> bool:
    text = norm(hit["text"])
    for g in _golds(q):
        if hit["document_id"] != g["source_id"]:
            continue
        if not any(hit["page"] <= p <= hit.get("page_end", hit["page"]) for p in g["pages"]):
            continue
        if norm(g["evidence"]) in text:
            return True
    return False


def partial_evidence(q: dict[str, Any], hit: dict[str, Any]) -> bool:
    """Right page and at least the first half of the evidence present: evidence split across chunks."""
    text = norm(hit["text"])
    for g in _golds(q):
        ev = norm(g["evidence"])
        half = ev[: max(20, len(ev) // 2)]
        if hit["document_id"] == g["source_id"] and (half in text or ev[-max(20, len(ev) // 2):] in text):
            return True
    return False


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    flags = [is_hit(question, h) for h in hits]
    return {
        "id": question["id"], "split": question["split"], "question": question["question"],
        "filter": question.get("filter"), "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"], "evidence_quote": question["evidence_quote"],
        "hit_at_1": bool(flags[:1] and flags[0]), "hit_at_3": any(flags[:3]),
        "evidence_split": (not any(flags[:3])) and any(partial_evidence(question, h) for h in hits[:3]),
        "filter_respected": (all(all(h.get(k) == v for k, v in question["filter"].items()) for h in hits[:3])
                             if question.get("filter") else None),
        "top3": [{"rank": i + 1, "chunk_id": h["chunk_id"], "document_id": h["document_id"], "source": h["source"],
                  "page": h["page"], "page_end": h.get("page_end"), "part": h.get("part"), "chapter": h.get("chapter"),
                  "article": h.get("article"), "annex": h.get("annex"), "score": round(float(h["score"]), 4),
                  "is_hit": flags[i], "text": h["text"]} for i, h in enumerate(hits[:3])],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in traces:
        key = (t["strategy"], t["split"])
        s = out.setdefault(f"{key[0]}/{key[1]}", {"strategy": key[0], "split": key[1], "n": 0, "hit1": 0, "hit3": 0,
                                                 "evidence_split": 0, "filter_q": 0, "filter_ok": 0})
        s["n"] += 1
        s["hit1"] += t["hit_at_1"]
        s["hit3"] += t["hit_at_3"]
        s["evidence_split"] += t["evidence_split"]
        if t["filter_respected"] is not None and t.get("filter_applied"):
            s["filter_q"] += 1
            s["filter_ok"] += t["filter_respected"]
    for s in out.values():
        s["hit1_str"] = f"{s['hit1']}/{s['n']}"
        s["hit3_str"] = f"{s['hit3']}/{s['n']}"
        s["filter_str"] = f"{s['filter_ok']}/{s['filter_q']}" if s["filter_q"] else "N/A"
    return out
