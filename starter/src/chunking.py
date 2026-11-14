"""Three chunking strategies over the same normalized pages.

A  fixed_size      : character windows inside one page, with overlap (ignores structure)
D  recursive       : split by paragraph -> line -> sentence -> word until <= max_size, inside one page
B  structure_aware : legal units (recital / Article / numbered paragraph / annex point), across pages,
                     packed up to max_size and prefixed with a short context header

Every chunk keeps document_id, source, page (start) and metadata page_end + structural fields.
"""
from __future__ import annotations

import hashlib
import re

from .ingestion import _ANNEX, _ARTICLE, _CHAPTER, _ENACTING
from .models import Chunk, DocumentPage


def _cid(*parts: object) -> str:
    return ":".join(str(p) for p in parts)


def _meta(page: DocumentPage, **extra) -> dict:
    m = {k: v for k, v in page.metadata.items() if k != "char_count"}
    m.update(extra)
    return m


# ---------------------------------------------------------------- A: fixed size
def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text, chunks, start, i = page.text, [], 0, 0
    while start < len(text):
        piece = text[start:start + size]
        chunks.append(Chunk(chunk_id=_cid("A", page.document_id, f"p{page.page}", i), document_id=page.document_id,
                            source=page.source, page=page.page, text=piece,
                            metadata=_meta(page, page_end=page.page, char_start=start)))
        if start + size >= len(text):
            break
        start += size - overlap
        i += 1
    return chunks


# ---------------------------------------------------------------- D: recursive
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, max_size: int, seps: list[str]) -> list[str]:
    if len(text) <= max_size:
        return [text]
    if not seps:
        return [text[i:i + max_size] for i in range(0, len(text), max_size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out, buf = [], ""
    for part in parts:
        candidate = (buf + sep + part) if buf else part
        if len(candidate) <= max_size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(part) > max_size:
            out.extend(_recursive_split(part, max_size, rest))
            buf = ""
        else:
            buf = part
    if buf:
        out.append(buf)
    return out


def recursive(page: DocumentPage, max_size: int, overlap: int) -> list[Chunk]:
    pieces = _recursive_split(page.text, max_size, _SEPARATORS)
    chunks = []
    for i, piece in enumerate(pieces):
        if overlap and i > 0:
            piece = pieces[i - 1][-overlap:] + " " + piece   # carry a little context forward
        chunks.append(Chunk(chunk_id=_cid("D", page.document_id, f"p{page.page}", i), document_id=page.document_id,
                            source=page.source, page=page.page, text=piece.strip(),
                            metadata=_meta(page, page_end=page.page)))
    return chunks


# ---------------------------------------------------------------- B: structure aware
_RECITAL = re.compile(r"^\((\d+)\)\s")
_PARA = re.compile(r"^(\d+)\.\s")
_POINT_ANNEX = re.compile(r"^(\d+)\.\s|^\([a-z]\)\s")


def _sentences(text: str, max_size: int) -> list[str]:
    return _recursive_split(text, max_size, [". ", " "])


def structure_aware(pages: list[DocumentPage], max_size: int) -> list[Chunk]:
    """Walk the document as one stream, start a new unit at each legal boundary, pack units."""
    if not pages:
        return []
    doc, source = pages[0].document_id, pages[0].source
    base = {k: v for k, v in pages[0].metadata.items() if k in ("regulation", "regulation_short", "year", "document_type")}
    state = {"part": "recital", "chapter": None, "article": None, "article_title": None, "annex": None}
    units: list[dict] = []
    cur: dict | None = None
    expect_title = False

    def start_unit(page_no: int, kind: str):
        nonlocal cur
        cur = {"lines": [], "page": page_no, "page_end": page_no, "kind": kind, **{k: state[k] for k in state}}
        units.append(cur)

    for p in pages:
        for line in p.text.split("\n"):
            boundary = None
            if _ENACTING.match(line):
                state["part"] = "article"
            if m := _CHAPTER.match(line):
                state.update(part="article", chapter=m.group(1), annex=None)
                boundary = "chapter"
            elif m := _ARTICLE.match(line):
                inline = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                state.update(part="article", article=m.group(1), annex=None, article_title=inline)
                boundary, expect_title = "article", inline is None
            elif m := _ANNEX.match(line):
                state.update(part="annex", annex=m.group(1), article=None, article_title=None)
                boundary = "annex"
            elif state["part"] == "recital" and _RECITAL.match(line):
                boundary = "recital"
            elif state["part"] == "article" and _PARA.match(line):
                boundary = "paragraph"
            elif state["part"] == "annex" and _POINT_ANNEX.match(line) and cur and len("\n".join(cur["lines"])) > max_size // 2:
                boundary = "annex_point"
            elif expect_title and state["article"]:
                state["article_title"] = line[:120]
                expect_title = False
                if cur is not None:
                    cur["article_title"] = state["article_title"]
            if boundary or cur is None:
                start_unit(p.page, boundary or "text")
            cur["lines"].append(line)
            cur["page_end"] = p.page

    # pack small consecutive units of the same article/annex/recital-block; split oversized ones
    chunks: list[Chunk] = []
    buf: dict | None = None

    def header(u: dict) -> str:
        reg = base.get("regulation_short", doc)
        if u["part"] == "annex" and u["annex"]:
            return f"[{reg} | Annex {u['annex']}]"
        if u["part"] == "article" and u["article"]:
            t = f" {u['article_title']}" if u.get("article_title") else ""
            ch = f" | Chapter {u['chapter']}" if u["chapter"] else ""
            return f"[{reg}{ch} | Article {u['article']}{t}]"
        return f"[{reg} | Recitals]"

    def emit(u: dict, text: str):
        n = len(chunks)
        chunks.append(Chunk(chunk_id=_cid("B", doc, n), document_id=doc, source=source, page=u["page"],
                            text=f"{header(u)}\n{text}".strip(),
                            metadata={**base, "part": u["part"], "chapter": u["chapter"], "article": u["article"],
                                      "annex": u["annex"], "page_end": u["page_end"], "unit": u["kind"]}))

    def same_scope(a: dict, b: dict) -> bool:
        return (a["part"], a["article"], a["annex"]) == (b["part"], b["article"], b["annex"])

    for u in units:
        text = "\n".join(u["lines"]).strip()
        if not text:
            continue
        if len(text) > max_size:
            if buf:
                emit(buf, buf["text"]); buf = None
            for piece in _sentences(text, max_size):
                emit(u, piece)
            continue
        if buf and same_scope(buf, u) and len(buf["text"]) + 1 + len(text) <= max_size:
            buf["text"] += "\n" + text
            buf["page_end"] = u["page_end"]
        else:
            if buf:
                emit(buf, buf["text"])
            buf = {**u, "text": text}
    if buf:
        emit(buf, buf["text"])
    return chunks


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
