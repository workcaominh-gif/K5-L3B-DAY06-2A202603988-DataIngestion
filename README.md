# Day 06 · Data Ingestion — Cao Duy Minh (2A202603988, L3B)

Pipeline biến **EU AI Act + GDPR** (572 trang nội dung) thành kho tri thức có thể truy xuất, rồi so sánh
**6 chiến lược chunking + 1 cấu hình lọc metadata** trên **3 model embedding**, với cùng corpus, cùng vector store và cùng 24 câu hỏi.

- Báo cáo cá nhân: [`report/individual/2A202603988_CaoDuyMinh.md`](report/individual/2A202603988_CaoDuyMinh.md)
- Bảng kết quả: [`starter/results/summary.md`](starter/results/summary.md) · nhật ký thí nghiệm: [`artifacts/experiment_log.csv`](artifacts/experiment_log.csv)
- Demo HTML: [`demo/index.html`](demo/index.html) (kết quả, so sánh từng câu hỏi, tìm thử trực tiếp)
- Đề bài gốc: [`docs/LAB_README.md`](docs/LAB_README.md)

## Chiến lược

| Mã | Chunking | Tham số | Lọc |
|---|---|---|---|
| F | Fixed-size, không overlap | 500 ký tự, theo trang | không |
| A | Fixed-size + overlap 10% | 500 / 50 | không |
| D | Recursive (đoạn → dòng → câu → từ) | ≤ 800 / overlap 80 | không |
| B | Theo cấu trúc văn bản luật (recital / Article / khoản / điểm Annex) + header ngữ cảnh | ≤ 1200, qua nhiều trang | không |
| S | Semantic: cắt ở chỗ độ tương đồng giữa 2 câu liền kề thấp nhất (phân vị 10) | 200–1200 | không |
| L | Agentic / LLM: Claude Haiku 5.5 chọn điểm cắt trên **30 trang mẫu** (25 trang gold + 5 ngẫu nhiên), các trang khác dùng D | ≤ 1200 | không |
| C | Giống B | | `part` = recital / article / annex, chỉ với 4 câu có ngữ cảnh lọc |

Cặp so sánh đổi đúng một yếu tố: F→A (overlap), A→D, D→B, B→S, D→L (cách cắt), B→C (lọc). Model embedding:
`BAAI/bge-small-en-v1.5` (384, model chính), `sentence-transformers/all-MiniLM-L6-v2` (384), `google/embeddinggemma-300m` (768).

## Cài đặt và chạy

Cần Docker, Python 3.12 (3.11–3.14), ~2 GB đĩa cho thư viện + model.

```bash
cd starter
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt "sentence-transformers>=5.1" "transformers>=4.56" "anthropic>=1"
cp .env.example .env        # DATABASE_URL; thêm HF_TOKEN (EmbeddingGemma là model gated) và ANTHROPIC_API_KEY (chỉ cho L)
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Đặt 2 PDF vào `starter/data/raw/` theo tên trong [`data/manifest.json`](data/manifest.json) (không commit vì PDF GDPR là bản dàn trang của bên thứ ba), rồi:

```bash
python -m src.run inspect          # đọc + chuẩn hoá -> data/processed/pages.jsonl, results/corpus_stats.json
python ../scripts/verify_benchmark.py   # 24/24 evidence nằm đúng trang gold
python -m src.run llm              # Haiku chọn điểm cắt trên 30 trang mẫu (cache kết quả + prompt caching)
python -m src.run ingest           # 6 chunking × 3 model -> pgvector; nạp lại lần 2 để kiểm tra không nhân đôi
python -m src.run eval             # top-3 cho 24 câu × 7 cấu hình × 3 model -> results/traces/<model>/
python -m src.report               # results/summary.md + results/report_data.json
python -m src.embspace             # results/embspace/<question>.json (embedding space + chunk page map in the demo)
python -m src.demo_server          # http://127.0.0.1:8770/demo/index.html
```

`python -m src.run ingest bge` / `eval bge` chạy riêng một model. `data/processed/llm_boundaries.json` đã có sẵn nên chạy lại L không tốn tiền.

## Dữ liệu nặng nằm ngoài repo

Trên máy tác giả, `.venv` (torch) và cache model Hugging Face nằm trên ổ ngoài: `starter/.venv` là symlink tới
`/Volumes/T7 Dev/AI20K/day6/venv`, `HF_HOME` trỏ tới `/Volumes/T7 Dev/AI20K/day6/hf_cache` (khai báo trong `starter/.env`).
Trên máy khác chỉ cần tạo `.venv` bình thường và bỏ dòng `HF_HOME`.

## Cấu trúc

```text
STUDENT.md · README.md · configs/strategies.json · data/manifest.json
benchmark/questions.jsonl (24 câu, 18 dev / 6 test) · benchmark/README.md (cách chia, tiêu chí chọn chốt trước)
starter/src/      ingestion, chunking (F A D B S L), llm_chunker, embeddings, store (pgvector), evaluation, run, report, embspace, demo_server
starter/data/processed/  pages.jsonl (văn bản đã chuẩn hoá theo trang), llm_boundaries.json
starter/results/  corpus_stats, extraction_errors, ingest_report, summary.json/.md, traces/<model>/traces_<S>.jsonl,
                  manual/ (coherence_labels.csv, metadata_audit_test.csv, grounding_labels_test.csv)
artifacts/experiment_log.csv · report/individual/2A202603988_CaoDuyMinh.md · demo/index.html
```

Không có secret trong repo: `.env` bị `.gitignore`; PDF gốc không commit.
