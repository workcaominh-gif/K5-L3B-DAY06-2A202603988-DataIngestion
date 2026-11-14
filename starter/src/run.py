"""Day 06 pipeline entry point.

    python -m src.run inspect   # read + normalize PDFs -> data/processed/pages.jsonl, extraction log
    python -m src.run llm       # Claude Haiku 5.5 picks chunk boundaries on a small page sample (cached)
    python -m src.run ingest [model]   # chunk F, A, D, B, S, L; embed with each model; upsert; re-ingest check
    python -m src.run eval [model]     # benchmark F, A, D, B, S, L and C (= B + metadata filter) per model

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
MIN_CONTENT_CHARS = 300


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
        for p in doc_pages:                        # rule from data/manifest.json
            p.metadata["counted_as_content"] = p.metadata["char_count"] >= MIN_CONTENT_CHARS
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
    stats = {"run_at": _now(), "rule": f">= {MIN_CONTENT_CHARS} normalized chars", "documents": {}}
    for d, ps in by_doc.items():
        counted = [p for p in ps if p.metadata["counted_as_content"]]
        stats["documents"][d] = {"pages_with_text": len(ps), "content_pages": len(counted),
                                 "content_page_ids": [p.page for p in counted],
                                 "excluded_page_ids": [p.page for p in ps if not p.metadata["counted_as_content"]],
                                 "chars": sum(len(p.text) for p in ps)}
        parts = {k: sum(1 for p in ps if p.metadata['part'] == k) for k in ("recital", "article", "annex")}
        print(f"  {d}: {len(ps)} pages, counted as content {len(counted)}, chars {sum(len(p.text) for p in ps)}, parts {parts}")
    (RESULTS / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


STRATEGIES = ["F", "A", "D", "B", "S", "L"]          # chunkings; C = index B + metadata filter at query time


def load_pages() -> list[DocumentPage]:
    rows = [json.loads(l) for l in (PROCESSED / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
    return [DocumentPage(r["document_id"], r["source"], r["page"], r["text"], r["metadata"]) for r in rows]


def gold_pages() -> set[tuple[str, int]]:
    out = set()
    for q in load_questions():
        out.add((q["gold_source_id"], q["gold_page"]))
        for a in q.get("alternatives", []):
            out.update((a["source_id"], p) for p in a["pages"])
    return out


def cmd_llm() -> None:
    """Agentic chunking boundaries from Claude Haiku 5.5 on a SMALL SAMPLE (gold pages + random pages)."""
    from . import llm_chunker
    cfg = load_config()["L"]
    pages = load_pages()
    sample = llm_chunker.sample_pages(pages, gold_pages(), cfg["random_pages"], cfg["seed"])
    res = llm_chunker.run(sample, PROCESSED / "llm_boundaries.json")
    print(json.dumps(res["summary"], indent=1))


def build_chunks(pages: list[DocumentPage], cfg: dict, sem_embed=None) -> dict[str, list[Chunk]]:
    from .chunking import fixed_no_overlap, llm_chunks, semantic
    by_doc: dict[str, list[DocumentPage]] = {}
    for p in pages:
        by_doc.setdefault(p.document_id, []).append(p)
    out: dict[str, list[Chunk]] = {k: [] for k in STRATEGIES}
    llm = json.loads((PROCESSED / "llm_boundaries.json").read_text(encoding="utf-8"))["pages"] \
        if (PROCESSED / "llm_boundaries.json").exists() else {}
    llm_b = {k: v["starts"] for k, v in llm.items()}
    for doc_pages in by_doc.values():
        for p in doc_pages:
            out["F"].extend(fixed_no_overlap(p, cfg["F"]["chunk_size"]))
            out["A"].extend(fixed_size(p, cfg["A"]["chunk_size"], cfg["A"]["overlap"]))
            out["D"].extend(recursive(p, cfg["D"]["chunk_size"], cfg["D"]["overlap"]))
        out["B"].extend(structure_aware(doc_pages, cfg["B"]["chunk_size"]))
        if sem_embed is not None:
            out["S"].extend(semantic(doc_pages, sem_embed, cfg["S"]["max_size"], cfg["S"]["min_size"], cfg["S"]["percentile"]))
        sampled = [p for p in doc_pages if f"{p.document_id}:{p.page}" in llm_b]
        rest = [p for p in doc_pages if f"{p.document_id}:{p.page}" not in llm_b]
        out["L"].extend(llm_chunks(sampled, llm_b, cfg["L"]["max_size"]))
        for p in rest:                                   # pages outside the LLM sample: recursive fallback
            for c in recursive(p, cfg["D"]["chunk_size"], cfg["D"]["overlap"]):
                c.chunk_id = c.chunk_id.replace("D:", "L:", 1)
                c.metadata["llm_fallback"] = True
                out["L"].append(c)
    return out


def _stats(chunks: list[Chunk]) -> dict:
    lens = [len(c.text) for c in chunks]
    return {"chunks": len(chunks), "chars_total": sum(lens), "len_min": min(lens), "len_median": int(statistics.median(lens)),
            "len_mean": round(statistics.mean(lens), 1), "len_max": max(lens)}


def get_store(model_key: str, emb):
    from .store import PgVectorStore
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL missing: copy starter/.env.example to starter/.env")
    store = PgVectorStore(url, emb.dimensions, table=f"chunks_{model_key}")
    store.create_schema()
    return store


def _models(cfg: dict, only: str | None) -> list[dict]:
    return [m for m in cfg["models"] if not only or m["key"] == only]


def cmd_ingest(only: str | None = None) -> None:
    from .embeddings import MyEmbedder
    cfg = load_config()
    pages = load_pages()
    sem_model = MyEmbedder(cfg["S"]["boundary_model"])          # fixed boundary model for S, all eval models
    chunks = build_chunks(pages, cfg, sem_embed=sem_model.embed_documents)
    del sem_model
    report_path = RESULTS / "ingest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"models": {}}
    report["chunk_stats"] = {k: _stats(v) for k, v in chunks.items()}
    for m in _models(cfg, only):
        emb = MyEmbedder(m["name"])
        store = get_store(m["key"], emb)
        mrep = {"model": m["name"], "dimensions": emb.dimensions, "device": emb.device, "run_at": _now(),
                "vector_store": "PostgreSQL 16 + pgvector 0.8.7, exact cosine (no ANN index)", "cost_usd": 0.0,
                "strategies": {}}
        for name, cs in chunks.items():
            t0 = time.perf_counter()
            vecs = emb.embed_documents([c.text for c in cs])
            t_embed = time.perf_counter() - t0
            t1 = time.perf_counter()
            first = store.upsert(name, cs, vecs)
            t_store = time.perf_counter() - t1
            second = store.upsert(name, cs, vecs)               # same version again: must change nothing
            mrep["strategies"][name] = {"chunks": len(cs), "embed_seconds": round(t_embed, 1),
                                        "store_seconds": round(t_store, 1), "first_upsert": first,
                                        "reingest_same_version": second, "rows_in_store": store.count(name)}
            print(f"[{m['key']}] {name}: {len(cs)} chunks, embed {t_embed:.1f}s, rows {store.count(name)}, re-ingest {second}")
        report["models"][m["key"]] = mrep
        RESULTS.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions() -> list[dict]:
    return [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_eval(only: str | None = None) -> None:
    from .embeddings import MyEmbedder
    cfg = load_config()
    qs = load_questions()
    runs = [(s, s, False) for s in STRATEGIES] + [("C", "B", True)]
    all_summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8")) if (RESULTS / "summary.json").exists() else {}
    for m in _models(cfg, only):
        emb = MyEmbedder(m["name"])
        store = get_store(m["key"], emb)
        out_dir = RESULTS / "traces" / m["key"]
        out_dir.mkdir(parents=True, exist_ok=True)
        traces = []
        t0 = time.perf_counter()
        qvecs = {q["id"]: emb.embed_query(q["question"]) for q in qs}
        for label, index, use_filter in runs:
            with open(out_dir / f"traces_{label}.jsonl", "w", encoding="utf-8") as f:
                for q in qs:
                    flt = q.get("filter") if use_filter else None
                    hits = store.search(index, qvecs[q["id"]], top_k=cfg["top_k"], filters=flt)
                    tr = evaluate_query(q, hits)
                    tr.update(model=m["key"], strategy=label, index=index, filter_applied=bool(flt))
                    traces.append(tr)
                    f.write(json.dumps(tr, ensure_ascii=False) + "\n")
        summ = summarize(traces)
        summ["_meta"] = {"run_at": _now(), "model": m["name"], "questions": len(qs),
                         "eval_seconds": round(time.perf_counter() - t0, 1)}
        all_summary[m["key"]] = summ
        print(f"\n== {m['key']} ({m['name']})")
        for k, s in summ.items():
            if not k.startswith("_"):
                print(f"  {s['strategy']} {s['split']:4} Hit@1 {s['hit1_str']:>6}  Hit@3 {s['hit3_str']:>6}  "
                      f"split-ev {s['evidence_split']}  filter {s['filter_str']}")
    (RESULTS / "summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    only = sys.argv[2] if len(sys.argv) > 2 else None
    {"inspect": cmd_inspect, "llm": cmd_llm, "ingest": lambda: cmd_ingest(only), "eval": lambda: cmd_eval(only)}.get(
        cmd, lambda: print(__doc__))()


if __name__ == "__main__":
    main()
