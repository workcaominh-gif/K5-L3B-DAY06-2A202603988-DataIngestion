"""Precompute the 'embedding space' view for the demo: for every model x strategy x question,
the 40 chunks nearest to the query plus every chunk that holds the full evidence.

    python -m src.embspace          # writes results/embspace/<question>.json  {model: {strategy: ...}}

Point = [rank, cosine, angle, status(2 full evidence / 1 part of it / 0 other), doc, pages, where, preview,
         length, cut_by_llm (L only), has_context_header].
Layout (honest about ranking): the query sits at the centre; each chunk's distance from the centre is
its true cosine distance (1 - cosine similarity) to the query, so ring order == retrieval order.
Only the angle is a projection: PCA (2 components) of the chunk vectors after removing their
component along the query, i.e. chunks that point the same "other" way sit near each other.
C uses index B with the question's part filter, exactly as in the benchmark.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv

from .embeddings import MyEmbedder
from .evaluation import is_hit, partial_evidence

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "results" / "embspace"
NEAR = 40


def _where(r: dict) -> str:
    if r["part"] == "recital":
        return "Recital"
    if r["annex"]:
        return f"Phụ lục {r['annex']}"
    if r["article"]:
        return f"Điều {r['article']}"
    return r["part"] or ""


def _n(t: str) -> str:
    return " ".join(t.split()).lower()


def _body(text: str) -> str:
    t = text.lstrip()
    return t.split("]", 1)[1] if t.startswith("[") and "]" in t[:200] else t   # drop B's context header


def _window(pages: dict, q: dict) -> dict:
    """Gold page +-1 as one normalised string, with page offsets and the evidence span."""
    doc, g = q["gold_source_id"], q["gold_page"]
    segs, T, pos = [], "", 0
    for pg in (g - 1, g, g + 1):
        t = pages.get((doc, pg))
        if t is None:
            continue
        t = _n(t)
        segs.append({"page": pg, "start": len(T), "end": len(T) + len(t)})
        T += t + " "
    ev = _n(q["evidence_quote"])
    i = T.find(ev)
    if i < 0:
        i = T.find(ev[:30])
    return {"doc": doc, "pages": segs, "len": len(T), "ev": [i, i + len(ev)] if i >= 0 else None, "_T": T}


def _locate(T: str, text: str) -> tuple[int, int] | None:
    b = _n(_body(text))
    if len(b) < 20:
        return None
    s = T.find(b[:60])
    if s < 0:
        return None
    e = T.find(b[-60:], s)
    return (s, e + 60) if e >= 0 else (s, s + len(b))


def main() -> None:
    load_dotenv(ROOT / ".env")
    cfg = json.loads((REPO / "configs/strategies.json").read_text(encoding="utf-8"))
    qs = [json.loads(l) for l in (REPO / "benchmark/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    llm_pages = set(json.loads((ROOT / "data/processed/llm_boundaries.json").read_text(encoding="utf-8"))["pages"])
    ALL: dict = {q["id"]: {} for q in qs}
    pages = {(r["document_id"], r["page"]): r["text"] for r in
             (json.loads(l) for l in (ROOT / "data/processed/pages.jsonl").read_text(encoding="utf-8").splitlines())}
    WIN = {q["id"]: _window(pages, q) for q in qs}
    PM: dict = {q["id"]: {"window": {k: v for k, v in WIN[q["id"]].items() if k != "_T"}, "chunks": {}, "cos": {}} for q in qs}
    for m in cfg["models"]:
        emb = MyEmbedder(m["name"])
        qv = {q["id"]: np.array(emb.embed_query(q["question"]), dtype=np.float32) for q in qs}
        with psycopg.connect(os.environ["DATABASE_URL"], row_factory=psycopg.rows.dict_row) as c:
            for label in ["F", "A", "D", "B", "S", "L", "C"]:
                index = "B" if label == "C" else label
                rows = c.execute(f"""SELECT chunk_id, document_id, page, page_end, part, article, annex, text,
                                     embedding::text AS e FROM chunks_{m['key']} WHERE strategy=%s""", (index,)).fetchall()
                V = np.array([json.loads(r["e"]) for r in rows], dtype=np.float32)
                V /= np.linalg.norm(V, axis=1, keepdims=True)
                for q in qs:
                    flt = (q.get("filter") or {}).get("part") if label == "C" else None
                    ok = np.array([flt is None or r["part"] == flt for r in rows])
                    sims = V @ qv[q["id"]]
                    sims_f = np.where(ok, sims, -9)
                    order = np.argsort(-sims_f)
                    rank = np.empty(len(rows), int)
                    rank[order] = np.arange(1, len(rows) + 1)
                    hits = [i for i in range(len(rows)) if ok[i] and is_hit(q, rows[i])]
                    parts = [i for i in range(len(rows)) if ok[i] and i not in hits and partial_evidence(q, rows[i])]
                    pick = list(dict.fromkeys(list(order[:NEAR]) + hits + parts))
                    X = V[pick] - np.outer(V[pick] @ qv[q["id"]], qv[q["id"]])   # drop the query direction
                    X -= X.mean(0)
                    _, _, vt = np.linalg.svd(X, full_matrices=False)
                    P = X @ vt[:2].T
                    ang = np.arctan2(P[:, 1], P[:, 0])
                    pts = []
                    for k, i in enumerate(pick):
                        r = rows[i]
                        pg = f"{r['page']}" if r["page_end"] in (None, r["page"]) else f"{r['page']}–{r['page_end']}"
                        status = 2 if i in hits else 1 if i in parts else 0     # 2 full evidence, 1 part of it, 0 other
                        llm = int(label == "L" and f"{r['document_id']}:{r['page']}" in llm_pages)
                        pts.append([int(rank[i]), round(float(sims[i]), 4), round(float(ang[k]), 3), status,
                                    "GDPR" if r["document_id"].startswith("gdpr") else "AI Act", pg, _where(r),
                                    " ".join(r["text"].split())[:220], len(r["text"]), llm,
                                    int(r["text"].lstrip().startswith("["))])
                    ALL[q["id"]].setdefault(m["key"], {})[label] = {"n": int(ok.sum()), "filter": flt, "pts": pts}
                    # page map: every chunk overlapping the gold page +-1, its span on those pages and its cosine
                    w, pm = WIN[q["id"]], PM[q["id"]]
                    pgs = [x["page"] for x in w["pages"]]
                    if label not in pm["chunks"]:
                        spans = []
                        for i, r in enumerate(rows):
                            if r["document_id"] != w["doc"] or not pgs:
                                continue
                            if (r["page_end"] or r["page"]) < pgs[0] or r["page"] > pgs[-1]:
                                continue
                            loc = _locate(w["_T"], r["text"])
                            if loc:
                                spans.append((loc[0], loc[1], r["chunk_id"], bool(ok[i])))
                        spans.sort()
                        pm["chunks"][label] = [[a, b, c, int(o)] for a, b, c, o in spans]
                    idx = {r["chunk_id"]: i for i, r in enumerate(rows)}
                    pm["cos"].setdefault(m["key"], {})[label] = [
                        [round(float(sims[idx[c]]), 4), int(rank[idx[c]]) if o else 0, 2 if idx[c] in hits else 1 if idx[c] in parts else 0]
                        for _, _, c, o in pm["chunks"][label]]
                print(m["key"], label, len(rows), "chunks")
    for f in OUT.glob("*.json"):
        f.unlink()
    for qid, d in ALL.items():
        d["_pagemap"] = PM[qid]
        (OUT / f"{qid}.json").write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
