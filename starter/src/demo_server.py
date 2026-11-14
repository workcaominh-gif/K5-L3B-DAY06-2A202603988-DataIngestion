"""Local demo server for demo/index.html.

    python -m src.demo_server            # http://127.0.0.1:8770

Serves the repo's demo/, benchmark/ and starter/results/ as static files, plus
    POST /api/search  {"query": "...", "model": "bge|minilm|gemma", "strategy": "F|A|D|B|S|L|C", "part": "recital|article|annex|"}
which embeds the query with the chosen model and searches the same pgvector tables the benchmark used.
Listens on 127.0.0.1 only. Without a running database the static tabs still work.
"""
from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
load_dotenv(ROOT / ".env")
PORT = int(os.environ.get("DEMO_PORT", "8770"))
_models: dict = {}


def _config() -> dict:
    return json.loads((REPO / "configs" / "strategies.json").read_text(encoding="utf-8"))


def search(body: dict) -> dict:
    from .embeddings import MyEmbedder
    from .store import PgVectorStore

    cfg = _config()
    key = body.get("model", "bge")
    m = next((m for m in cfg["models"] if m["key"] == key), None)
    if m is None:
        raise ValueError(f"unknown model {key}")
    strategy = body.get("strategy", "B")
    if strategy not in {"F", "A", "D", "B", "S", "L", "C"}:
        raise ValueError("unknown strategy")
    query = str(body.get("query", "")).strip()[:500]
    if not query:
        raise ValueError("empty query")
    if key not in _models:
        _models[key] = MyEmbedder(m["name"])
    emb = _models[key]
    store = PgVectorStore(os.environ["DATABASE_URL"], emb.dimensions, table=f"chunks_{key}")
    part = body.get("part") or None
    filters = {"part": part} if (part and strategy == "C") else None
    index = "B" if strategy == "C" else strategy
    hits = store.search(index, emb.embed_query(query), top_k=3, filters=filters)
    for h in hits:
        h["score"] = round(float(h["score"]), 4)
    return {"model": m["name"], "strategy": strategy, "filter": filters, "hits": hits}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(REPO), **k)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/demo/index.html")
            self.end_headers()
            return
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/search":
            return self.send_error(404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            code, payload = 200, search(body)
        except Exception as exc:  # report the real error; never fall back to fake results
            code, payload = 400, {"error": f"{type(exc).__name__}: {exc}"}
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"Demo on http://127.0.0.1:{PORT}/demo/index.html")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
