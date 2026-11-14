# Benchmark — 24 questions (18 dev / 6 test)

Written and verified **before** any retrieval run (see git history: this file is committed before `results/`).

- File: `questions.jsonl`. Fields: `id, split, type, question, expected_answer, gold_source_id, gold_page, evidence_quote, filter?, alternatives?`.
- `gold_page` = physical PDF page number in the file listed in `data/manifest.json`.
- Every `evidence_quote` was checked to occur on its gold page after the same normalization the pipeline uses
  (`python scripts/verify_benchmark.py` → 24/24).
- Types: direct 10, paraphrase 4, condition 3, exception 2, filter 4, version 1 (q23: the EP adopted text has a
  placeholder date instead of 2 August 2026 — a deliberate "which version am I reading" trap).
- q01 has two valid gold locations (Article 33 on p71 and recital 85 on p24); either counts as a hit.

## Split

Fixed by hand before running anything: test = q04, q09, q14, q19, q21, q23 — both documents, one filter question
(q19), and one of each harder type (paraphrase, condition, exception, version). The rest are dev.
**Leakage risk:** dev and test come from the same two documents and some test answers sit in the same Articles'
neighbourhood as dev answers (e.g. GDPR Chapter III); the test set is small (6) and only illustrative.

## Metadata filter questions

`filter` is supplied by the question's context *before* retrieval and never names the gold document or page:
q08, q20 → `{"part": "recital"}` ("according to the recitals …"); q19, q24 → `{"part": "annex"}` ("within the
annexes …"). Strategy C applies the filter only to these 4 questions and runs unfiltered on the other 20.

## Strategy selection criterion (pre-registered, decided before looking at test)

1. Highest dev **Evidence Hit@3**; 2. tie → higher dev **Hit@1**; 3. tie → higher coherence (manual);
4. tie → fewer chunks / lower ingest time. The chosen strategy is then reported on test once, without changes.
