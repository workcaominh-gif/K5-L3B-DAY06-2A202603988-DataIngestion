"""Read and normalize source PDFs, keeping a traceable link back to document + physical page.

Each physical page becomes one DocumentPage. Metadata comes only from the source text itself:
- regulation: fixed per document from the manifest (AI Act / GDPR)
- part: recital | article | annex, tracked by a running state machine over the page stream
- chapter / article / annex: the heading in force at the START of the page
Pages with no extractable text are recorded in the error log, never silently dropped.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

# Official Journal running headers / footers that repeat on every page (not content).
_NOISE_LINES = [
    re.compile(r"^EN\s*$"),
    re.compile(r"^OJ L,?\s*[\d.]+\s*$"),
    re.compile(r"^OJ L \d+/\d+\s*$"),
    re.compile(r"^L \d+/\d+\s+EN\s+Official Journal of the European Union\s+[\d.]+\s*$"),
    re.compile(r"^[\d.]+\s+EN\s+Official Journal of the European Union\s+L \d+/\d+\s*$"),
    re.compile(r"^Official Journal of the European Union\s*$"),
    re.compile(r"^ELI:\s*http\S+\s*$"),
    re.compile(r"^\d+/\d+\s*$"),            # "12/144" page counters
    re.compile(r"^EN\s+\d+/\d+\s*$"),
]

_CHAPTER = re.compile(r"^(?:CHAPTER|Chapter)\s+([IVXLC]+)\b")
# "Article 60" (title on next line, EP text) or "Article 16 Right to rectification" (title inline, GDPR);
# lines like "Article 5 shall apply ..." are cross-references, not headings.
_ARTICLE = re.compile(r"^Article\s+(\d+[a-z]?)(?:\s+([A-Z][^.]{2,110}))?\s*$")
_ANNEX = re.compile(r"^ANNEX\s+([IVXLC]+)\b")
_ENACTING = re.compile(r"^HAVE ADOPTED THIS REGULATION", re.I)


LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


def normalize_text(text: str) -> str:
    """NFC-normalize, repair PDF line-break hyphenation and spacing, keep paragraph breaks."""
    text = unicodedata.normalize("NFC", text)
    for lig, rep in LIGATURES.items():                       # NFC keeps ﬁ/ﬂ ligatures: expand them
        text = text.replace(lig, rep)
    text = text.replace("▌", "")                       # "▌" change bars in the EP text
    text = text.replace("­", "")                       # soft hyphen
    text = text.replace(" ", " ").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # word-\nbreak -> wordbreak
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not any(p.match(ln) for p in _NOISE_LINES)]
    # a bare number as the first or last line is a page number, not content
    if lines and re.fullmatch(r"\d{1,3}", lines[-1]):
        lines = lines[:-1]
    if lines and re.fullmatch(r"\d{1,3}", lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def load_pdf(path: Path, document_id: str, source: str, base_meta: dict | None = None,
             errors: list[dict] | None = None) -> list[DocumentPage]:
    """Extract every physical page; keep page number, source and structural metadata."""
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    state = {"part": "recital", "chapter": None, "article": None, "annex": None}
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # keep going; record the failure
            raw = ""
            if errors is not None:
                errors.append({"document_id": document_id, "page": i, "error": f"extract failed: {exc}"})
        text = normalize_text(raw)
        start_state = dict(state)
        for line in text.split("\n"):
            if _ENACTING.match(line):
                state["part"] = "article"
            elif m := _CHAPTER.match(line):
                state.update(part="article", chapter=m.group(1), annex=None)
            elif m := _ARTICLE.match(line):
                state.update(part="article", article=m.group(1), annex=None)
            elif m := _ANNEX.match(line):
                state.update(part="annex", annex=m.group(1), article=None)
        if not text:
            if errors is not None:
                errors.append({"document_id": document_id, "page": i, "error": "no extractable text"})
            continue
        meta = dict(base_meta or {})
        meta.update({
            "part": start_state["part"],
            "chapter": start_state["chapter"],
            "article": start_state["article"],
            "annex": start_state["annex"],
            "char_count": len(text),
        })
        pages.append(DocumentPage(document_id=document_id, source=source, page=i, text=text, metadata=meta))
    return pages
