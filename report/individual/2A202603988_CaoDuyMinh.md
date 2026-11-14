# Báo cáo cá nhân — Day 6 · Cao Duy Minh (2A202603988, L3B)

> Công cụ AI: Claude Code (Claude) hỗ trợ viết code, chạy pipeline, gán nhãn thủ công nháp và soạn báo cáo; học viên rà lại kết quả, nhãn và kết luận. Mọi số liệu dưới đây lấy từ file kết quả đã chạy (`starter/results/`), không ước lượng tay.

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:** quy định EU về AI và dữ liệu cá nhân.
  - `eu-ai-act-2024`: AI Act, văn bản Nghị viện châu Âu thông qua 13/03/2024 (P9_TA(2024)0138), 459 trang; tái sử dụng được nếu ghi nguồn (legal notice của EP).
  - `gdpr-2016`: GDPR, Regulation (EU) 2016/679, 117 trang, bản dàn trang LaTeX của bên thứ ba; nội dung là văn bản luật chính thức (tái sử dụng theo Decision 2011/833/EU), PDF không phân phối lại.
  - Quy tắc: một trang vật lý là "trang nội dung" khi văn bản sau chuẩn hoá có ≥ 300 ký tự → **456 + 116 = 572 trang** (loại 4 trang). Danh sách trang tính/loại: `starter/results/corpus_stats.json`. Nguồn, sha256, ngày truy cập: `data/manifest.json`.
  - EUR-Lex đang bảo trì lúc làm bài nên không lấy được bản Công báo; bản EP có ngày áp dụng là chỗ trống "… [24 months …]" (được dùng làm câu bẫy phiên bản q23).
- **Chuẩn hoá, lỗi và giới hạn:** pypdf 6.21, NFC; tách chữ ghép `ﬁ/ﬂ/ﬃ` (NFC không tách, làm "specification" không khớp), bỏ dấu gạch nối cuối dòng, bỏ ký tự `▌` của EP và số trang đứng riêng ở đầu/cuối trang. 0 trang lỗi trích xuất (`results/extraction_errors.json`). Giới hạn: không OCR (không cần); bảng/chú thích chân trang của EP lẫn vào văn bản; tiêu đề Điều có 2 dạng (EP: "Article 60" + tiêu đề dòng sau; GDPR: "Article 16 Right to rectification") nên regex phải xử lý cả hai.
- **Embedding / vector store / chi phí:**
  - Model chính `BAAI/bge-small-en-v1.5` (384 chiều, query có tiền tố hướng dẫn của BGE), so thêm `sentence-transformers/all-MiniLM-L6-v2` (384) và `google/embeddinggemma-300m` (768, prompt query/document sẵn của model). Chạy local bằng sentence-transformers 6.3 / torch 2.15 trên GPU Apple (MPS), vector chuẩn hoá L2.
  - PostgreSQL 16.16 + **pgvector 0.8.7** (Docker), mỗi model một bảng `chunks_<model>`, khoá `(strategy, chunk_id)`, **tìm cosine chính xác** (không dùng index ANN để không mất kết quả khi lọc), lọc metadata trong SQL trước khi xếp hạng.
  - Thời gian embed một chiến lược: bge ~6 s, MiniLM ~3–4 s, Gemma ~53–71 s (`results/ingest_report.json`). Chi phí embedding $0. LLM chunking (L): Claude Haiku 5.5, 30 trang, **≈ $0,0034** (+ $0,0006 lượt thử 2 trang), prompt caching: 34.590 token đọc từ cache.
  - Lệnh chạy: xem `README.md` (`python -m src.run inspect | llm | ingest | eval`, `python -m src.report`).
- **Metadata:** bắt buộc `document_id`, `source`, `page` (+ `page_end` cho chunk qua nhiều trang). Thêm `regulation`, `year`, `part` (recital / article / annex), `chapter`, `article`, `annex`, tất cả đọc từ chính văn bản (máy trạng thái theo tiêu đề), không đoán. **Trường lọc: `part`**, vì người hỏi luật thường chỉ định "theo phần giải thích (recital)" hay "trong phụ lục", và hai văn bản đều có các phần này với nội dung lặp ý nhau.

## Cấu hình và benchmark

- **Chiến lược** (cùng corpus, model, store, câu hỏi; `configs/strategies.json`):

  | | Chunking | Tham số | Lọc |
  |---|---|---|---|
  | F | Fixed-size, không overlap | 500 ký tự / trang | không |
  | A | Fixed-size + overlap 10% | 500 / 50 | không |
  | D | Recursive (đoạn → dòng → câu → từ) | ≤ 800 / overlap 80 | không |
  | B | Theo cấu trúc luật + header ngữ cảnh `[GDPR \| Chapter III \| Article 12 …]`, qua nhiều trang | ≤ 1200 | không |
  | S | Semantic: cắt ở 10% cặp câu liền kề có độ tương đồng thấp nhất (bge-small) | 200–1200 | không |
  | L | Agentic: Haiku 5.5 chọn dòng bắt đầu ý mới trên **30 trang mẫu** (25 trang gold + 5 ngẫu nhiên, seed 42); trang khác dùng D | ≤ 1200 | không |
  | C | Giống B | | `part`, chỉ cho 4 câu có ngữ cảnh lọc |

  Khác biệt duy nhất **B–C: lọc metadata** (cùng index B). Các cặp chunking đổi một yếu tố: F–A overlap, A–D/D–B/B–S/D–L cách cắt.
- **24 câu (18 dev / 6 test)**, chia tay trước khi chạy: test = q04, q09, q14, q19, q21, q23 (cả hai văn bản; 1 câu lọc; mỗi loại khó 1 câu). Loại: trực tiếp 10, diễn đạt lại 4, điều kiện 3, ngoại lệ 2, lọc 4 (q08, q20 `recital`; q19 test, q24 `annex`), phiên bản 1. Evidence được kiểm bằng `scripts/verify_benchmark.py`: **24/24** câu trích nằm đúng trang gold sau chuẩn hoá; q01 có 2 vị trí gold hợp lệ (Điều 33 và recital 85). Benchmark được commit (`ef5639f`) **trước** mọi lần truy xuất.
- **Tiêu chí chọn (chốt trước khi xem test, `benchmark/README.md`):** dev Evidence Hit@3 cao nhất → dev Hit@1 → coherence → ít chunk / ingest nhanh hơn. Áp dụng trên model chính bge.

## Kết quả

Model chính `bge-small-en-v1.5` (chi tiết 3 model: `starter/results/summary.md`, trace: `starter/results/traces/<model>/traces_<S>.jsonl`):

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian embed | Chi phí |
|---|---|---|---|---|---|---|
| F | dev / test | 3/18 · 1/6 | 6/18 · 2/6 | 2152 | 6.8 s | $0 |
| A | dev / test | 7/18 · 2/6 | 8/18 · 4/6 | 2283 | 7.5 s | $0 |
| D | dev / test | 8/18 · 2/6 | 9/18 · 4/6 | 1487 | 5.9 s | $0 |
| B | dev / test | 8/18 · 2/6 | 13/18 · 3/6 | 1212 | 6.0 s | $0 |
| S | dev / test | 6/18 · 2/6 | 8/18 · 3/6 | 1280 | 5.8 s | $0 (+ embed câu cho ranh giới) |
| L | dev / test | 9/18 · 2/6 | 13/18 · 3/6 | 1595 | 6.7 s | ≈ $0,0034 (Haiku, 30 trang) |
| **C** | dev / test | 8/18 · 2/6 | **14/18** · **4/6** | 1212 | 6.0 s | $0 |

So model (dev Evidence Hit@3, cùng chunk và câu hỏi): F 6/7/8 · A 8/9/8 · D 9/10/13 · **B 13/12/18** · S 8/9/14 · L 13/13/14 · **C 14/12/18** (bge / MiniLM / Gemma). Test Evidence Hit@3 của C: 4/6 · 3/6 · 4/6.

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| F | 0.1 (10) | 14/18 | N/A | 1/3/2 · 1.17 |
| A | 0.1 (10) | 12/18 | N/A | 1/1/4 · 1.5 |
| D | 0.8 (10) | 14/18 | N/A | 1/1/4 · 1.5 |
| B | 1.6 (10) | 18/18 | N/A | 1/1/4 · 1.5 |
| S | 1.6 (10) | 14/18 | N/A | 2/0/4 · 1.33 |
| L | 0.8 (10)* | 12/18 | N/A | 1/2/3 · 1.33 |
| C | 1.6 (10, = B) | 18/18 | **4/4** (dev 3/3, test 1/1) | 1/1/4 · 1.5 |

\* Không vị trí coherence nào rơi vào 30 trang mẫu của L, nên ở đó L chính là D.

**Nhãn và cách chọn mẫu:** `starter/results/manual/`.
- Coherence: 10 vị trí cố định chọn trước khi xem điểm, là giữa trang của 10 trang nội dung cách đều theo thứ tự; mỗi chiến lược lấy chunk chứa vị trí đó (`coherence_labels.csv`, có lý do).
- Metadata: 6 câu test × top-3. Kiểm `source` và `page` (chữ của chunk có trên trang ghi), `part` và `article` tại vị trí chunk bắt đầu, bằng cách quét lại cấu trúc; được tự động hoá rồi đọc lại (`metadata_audit_test.csv`).
- Grounding: 6 câu test × top-3, chấm 0/1/2 (`grounding_labels_test.csv`).
- Người chấm: Claude (trợ lý AI) theo thang của đề, học viên rà lại. Đây là nhãn phía tác giả, có nguy cơ thiên lệch.

## Phân tích và kết luận

- **A–B (thay chunking):** từ cắt cố định sang cắt theo cấu trúc, dev Hit@3 tăng 8 → 13/18 (bge), coherence 0,1 → 1,6, metadata 12 → 18/18, số chunk giảm 2283 → 1212. Thứ tự này đúng trên cả 3 model. Overlap riêng (F → A) đã cứu một phần: dev Hit@3 6 → 8, số `evidence_split` 2 → 1.
- **B–C (thay lọc):** lọc `part` chỉ chạy ở 4 câu; tuân thủ 4/4. Dev Hit@3 13 → 14 (q24 vào top-3), test 3 → 4: q19 không lọc trả về recital (56) (nói cùng ý nhưng không phải trang gold), có lọc `annex` thì lấy đúng Phụ lục III điểm 3 ở hạng 2. Không câu nào bị lọc làm xấu đi (bge: thêm q19, q24; MiniLM: thêm q19; Gemma: không đổi — q19 trượt ở cả B lẫn C, lọc `annex` cũng không đưa Phụ lục III vào top-3).
- **Ví dụ sai 1 — q23 (test, `eu-ai-act-2024` p416):** mọi chiến lược × mọi model trượt. Top-3 là trang bìa p1/p3 ("European Parliament … TEXTS ADOPTED") và Điều 49, vì câu hỏi có "adopted Parliament text", trùng từ vựng với trang bìa. Câu trả lời đúng là một chỗ trống "… [24 months …]" trong Điều 113. Đây đúng là lỗi phiên bản mà slide cảnh báo: retrieval không biết văn bản nào có hiệu lực; cần metadata `effective_date`/`version` và lọc theo phiên bản thay vì dựa vào embedding.
- **Ví dụ sai 2 — metadata cấp trang (q04/q09/q14, nhiều chiến lược):** F, A, D, S, L gắn `article` = Điều đang hiệu lực ở **đầu** trang, nên đoạn "72 hours" (GDPR p71, Điều 33) bị ghi Điều 32, đoạn định nghĩa AI system (p166) bị ghi Điều 2. "Đủ trường" nhưng sai, nên chỉ B/C (theo dõi cấu trúc từng dòng) đạt 18/18. Thêm: q04 với S trượt hẳn (grounding 0): top-3 là Điều 33 (72 giờ báo vi phạm), Điều 14 và recital 63, tức các đoạn "có thời hạn" khác; chunk chứa Điều 12(3) không lọt vào top-3.
- **Phân tích từng câu bằng demo (`src/embspace.py` + tab So sánh câu hỏi):** với mỗi câu, model và kỹ thuật, em tính hạng thật của chunk chứa bằng chứng trong cả index (không chỉ top-3), đối chiếu khớp 100% với trace benchmark. Hai điều rút ra:
  - Trượt có hai loại khác nhau: (1) không chunk nào chứa trọn câu (lỗi chunking, đổi model không cứu được, thấy rõ trên bản đồ chunk là ô viền vàng cắt ngang câu); (2) chunk đúng tồn tại nhưng bị đoạn cùng chủ đề vượt lên (q19: B xếp Phụ lục III #23 sau recital; lọc `annex` đưa lên #2, Gemma xếp #11).
  - Khi hai kỹ thuật cùng trúng #1, chênh cos chỉ 0,01–0,04 và không đổi kết quả; các yếu tố quan sát được (header, tỉ lệ câu bằng chứng trong chunk, chunk bắt đầu giữa câu) giải thích một phần nhưng không phải phép đo nhân quả.
- **Chiến lược chọn:** theo tiêu chí đã chốt, **C** (dev Hit@3 14/18 cao nhất với bge), báo cáo test một lần: Hit@1 2/6, Evidence Hit@3 4/6, grounding 1,5/2, metadata 18/18. Nếu được đổi model, **Gemma + B/C** tốt nhất (dev Hit@3 18/18, Hit@1 12/18; test 4/6), nhưng embed chậm hơn ~10 lần và nặng hơn (768 chiều, model 1,2 GB). Điều kiện áp dụng: văn bản có cấu trúc rõ (Điều/khoản/phụ lục); với văn bản phẳng không có tiêu đề, B mất lợi thế và D/S là lựa chọn rẻ hơn.
- **Hạn chế:**
  - Test chỉ 6 câu: chênh 1 câu là 17 điểm %, nên không kết luận tổng quát.
  - Gemma được so trên chính tập dev dùng để chọn chiến lược, nên kết quả dev 18/18 có thể lạc quan.
  - **L bị thiên lệch có lợi:** chỉ 30 trang (gồm toàn bộ trang gold) được LLM cắt, trang gây nhiễu giống D. L chỉ cho biết chất lượng cắt trên trang chứa đáp án, không phải hiệu quả nếu chạy toàn corpus.
  - Evidence Hit đòi chunk chứa **trọn** câu trích, nên chunk nhỏ (F/A) bị phạt khi câu trích dài; grounding thủ công cho thấy vài trường hợp đủ ý dù không trùng trang gold (q19 với B/S, recital 56).
  - Nhãn chấm tay do tác giả và trợ lý AI gán.
  - Thời gian đo trên một máy (Apple GPU), một lần chạy.
  - Bản AI Act là bản EP, không phải bản Công báo; bản GDPR là dàn trang của bên thứ ba.
- **Nếu làm lại / mở rộng:**
  1. Thêm metadata `version`/`effective_date` và lọc phiên bản (sửa q23).
  2. Chạy L trên mẫu ngẫu nhiên không chứa trang gold, hoặc toàn corpus, để so công bằng.
  3. Thêm hybrid BM25 + dense cho câu có số liệu (72 hours, 10^25, EUR 35 000 000).
  4. Gắn metadata theo vị trí chunk thay vì theo trang cho F/A/D/S.
  5. Viết thêm câu test và để người thứ hai chấm độc lập (tính độ đồng thuận).
