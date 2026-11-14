"""Day 06 pipeline entry point.

    python -m src.run inspect   # read + normalize PDFs -> data/processed/pages.jsonl, extraction log
    python -m src.run ingest    # chunk (A, D, B), embed, upsert into pgvector; re-ingest check
    python -m src.run eval      # run benchmark on A, D, B, C (= B + metadata filter) -> results/

Same corpus, same embedding model, same questions for every strategy.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .chunking import fixed_size, recursive, structure_aware
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage

ROOT = Path(__file__).resolve().parents[1]          # starter/
REPO = ROOT.parent
load_dotenv(ROOT / ".env")

MANIFEST = REPO / "data" / "manifest.json"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
BENCH = REPO / "benchmark" / "questions.jsonl"
CONFIG = REPO / "configs" / "strategies.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load_corpus(errors: list[dict]) -> list[DocumentPage]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pages: list[DocumentPage] = []
    for doc in manifest["documents"]:
        path = ROOT / doc["local_path"]
        if not path.exists():
            sys.exit(f"Missing source file {path}. See data/manifest.json for where to download it.")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if doc.get("sha256") and doc["sha256"] != sha:
            errors.append({"document_id": doc["source_id"], "page": None,
                           "error": f"sha256 differs from manifest ({sha[:12]}…): source changed"})
        base = {"regulation": doc["regulation"], "regulation_short": doc["regulation_short"], "year": doc["year"],
                "document_type": doc["document_type"]}
        doc_pages = load_pdf(path, doc["source_id"], doc["source_path_or_url"], base, errors)
        counted = set(doc.get("counted_pages", []))
        for p in doc_pages:
            p.metadata["counted_as_content"] = (p.page in counted) if counted else True
        pages.extend(doc_pages)
    return pages


def cmd_inspect() -> None:
    errors: list[dict] = []
    t0 = time.perf_counter()
    pages = load_corpus(errors)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "pages.jsonl", "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps({"document_id": p.document_id, "page": p.page, "source": p.source,
                                "metadata": p.metadata, "text": p.text}, ensure_ascii=False) + "\n")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "extraction_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    by_doc: dict[str, list[DocumentPage]] = {}
    for p in pages:
        by_doc.setdefault(p.document_id, []).append(p)
    print(f"read {len(pages)} pages with text in {time.perf_counter() - t0:.1f}s; {len(errors)} extraction issues")
    for d, ps in by_doc.items():
        counted = [p for p in ps if p.metadata["counted_as_content"]]
        parts = {k: sum(1 for p in ps if p.metadata['part'] == k) for k in ("recital", "article", "annex")}
        print(f"  {d}: {len(ps)} pages, counted as content {len(counted)}, chars {sum(len(p.text) for p in ps)}, parts {parts}")


def build_chunks(pages: list[DocumentPage], cfg: dict) -> dict[str, list[Chunk]]:
    by_doc: dict[str, list[DocumentPage]] = {}
    for p in pages:
        by_doc.setdefault(p.document_id, []).append(p)
    a, d, b = cfg["A"], cfg["D"], cfg["B"]
    out = {"A": [], "D": [], "B": []}
    for doc_pages in by_doc.values():
        for p in doc_pages:
            out["A"].extend(fixed_size(p, a["chunk_size"], a["overlap"]))
            out["D"].extend(recursive(p, d["chunk_size"], d["overlap"]))
        out["B"].extend(structure_aware(doc_pages, b["chunk_size"]))
    return out


def _stats(chunks: list[Chunk]) -> dict:
    lens = [len(c.text) for c in chunks]
    return {"chunks": len(chunks), "chars_total": sum(lens), "len_min": min(lens), "len_median": int(statistics.median(lens)),
            "len_mean": round(statistics.mean(lens), 1), "len_max": max(lens)}


def get_store_and_embedder(cfg: dict):
    from .embeddings import MyEmbedder
    from .store import PgVectorStore

    emb = MyEmbedder(cfg["embedding_model"])
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL missing: copy starter/.env.example to starter/.env")
    store = PgVectorStore(url, emb.dimensions)
    store.create_schema()
    return store, emb


def cmd_ingest() -> None:
    cfg = load_config()
    errors: list[dict] = []
    pages = load_corpus(errors)
    chunks = build_chunks(pages, cfg)
    store, emb = get_store_and_embedder(cfg)
    report = {"run_at": _now(), "embedding_model": emb.model_name, "dimensions": emb.dimensions,
              "vector_store": "PostgreSQL 16 + pgvector 0.8.7 (exact cosine, no ANN index)", "cost_usd": 0.0,
              "strategies": {}}
    for name, cs in chunks.items():
        t0 = time.perf_counter()
        vecs = emb.embed_documents([c.text for c in cs])
        t_embed = time.perf_counter() - t0
        t1 = time.perf_counter()
        first = store.upsert(name, cs, vecs)
        t_store = time.perf_counter() - t1
        second = store.upsert(name, cs, vecs)                 # re-ingest same version: must be a no-op
        report["strategies"][name] = {**_stats(cs), "embed_seconds": round(t_embed, 1),
                                      "store_seconds": round(t_store, 1), "first_upsert": first,
                                      "reingest_same_version": second, "rows_in_store": store.count(name)}
        print(f"{name}: {len(cs)} chunks, embed {t_embed:.1f}s, rows {store.count(name)}, re-ingest {second}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ingest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions() -> list[dict]:
    return [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_eval() -> None:
    cfg = load_config()
    store, emb = get_store_and_embedder(cfg)
    qs = load_questions()
    runs = [("A", "A", False), ("D", "D", False), ("B", "B", False), ("C", "B", True)]
    traces: list[dict] = []
    t0 = time.perf_counter()
    for label, index, use_filter in runs:
        with open(RESULTS / f"traces_{label}.jsonl", "w", encoding="utf-8") as f:
            for q in qs:
                qv = emb.embed_query(q["question"])
                flt = q.get("filter") if use_filter else None
                hits = store.search(index, qv, top_k=3, filters=flt)
                tr = evaluate_query(q, hits)
                tr.update(strategy=label, index=index, filter_applied=bool(flt))
                traces.append(tr)
                f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    summary = summarize(traces)
    summary["_meta"] = {"run_at": _now(), "embedding_model": emb.model_name, "questions": len(qs),
                        "eval_seconds": round(time.perf_counter() - t0, 1)}
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'strategy':8} {'split':5} {'Hit@1':>7} {'Hit@3':>7} {'split-ev':>8} {'filter':>7}")
    for k, s in summary.items():
        if k.startswith("_"):
            continue
        print(f"{s['strategy']:8} {s['split']:5} {s['hit1_str']:>7} {s['hit3_str']:>7} {s['evidence_split']:>8} {s['filter_str']:>7}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    {"inspect": cmd_inspect, "ingest": cmd_ingest, "eval": cmd_eval}.get(cmd, lambda: print(__doc__))()


if __name__ == "__main__":
    main()
