# Thuyết trình 2 phút với web demo

Mở `cd starter && python -m src.demo_server` rồi vào http://127.0.0.1:8770 trước khi nói.

## 0:00–0:20 · Bài toán (tab Kết quả)
"Em xây pipeline nạp dữ liệu cho RAG trên EU AI Act và GDPR: 572 trang, lưu vào PostgreSQL + pgvector.
Câu hỏi của bài: cắt văn bản thế nào và dùng model embedding nào thì tìm đúng đoạn bằng chứng nhất.
Em so 7 cách chunking × 3 model embedding trên 24 câu hỏi có đáp án và số trang (18 dev, 6 test)."

## 0:20–0:50 · Kết quả tổng (bảng heatmap)
"Thước đo là Evidence Hit@3: chunk trong top-3 phải chứa trọn câu trích dẫn, không chỉ đúng trang.
- Cắt cố định 500 ký tự (F) thấp nhất: 6/18 với bge.
- Cắt theo cấu trúc luật Điều/khoản, có header (B) tốt nhất: 13/18 với bge, 18/18 với Gemma.
- Thêm lọc metadata (C) cứu được câu hỏi về Phụ lục: bge 14/18, test 4/6. Em chọn C theo tiêu chí đặt trước.
- Haiku chunking (L) chỉ chạy 30 trang mẫu, tốn khoảng $0,004 nhờ cache system prompt, nên kết quả L hơi lạc quan."

## 0:50–1:30 · Một câu cụ thể (tab So sánh câu hỏi, chọn q19)
"Đây là cả 3 model cùng lúc, mỗi cột là một kỹ thuật, xếp hạng 1–2–3 theo vị trí chunk bằng chứng.
Bản đồ chunk phía trên cho thấy từng kỹ thuật cắt trang bằng chứng ra sao: khung xanh dương là câu bằng chứng,
viền xanh lá là chunk chứa trọn, viền vàng là chunk bị cắt ngang câu.
Với q19 chỉ C trúng: B có đúng chunk nhưng xếp #23 vì recital cùng chủ đề đứng trên; lọc annex đưa nó lên #2."

## 1:30–1:50 · Giải thích (bấm dấu ? ở một kỹ thuật)
"Dấu ? giải thích theo đúng câu, model và hạng hiện tại: so sánh các kỹ thuật hoặc các model cạnh nhau,
xem chính các chunk được tạo ra, không gian embedding (khoảng cách tới tâm = 1 − cos thật), và code chunking, embedding, truy vấn."

## 1:50–2:00 · Kết luận
"Kết luận: chunking theo cấu trúc văn bản quan trọng hơn đổi model; model tốt hơn (Gemma) cộng thêm,
còn lọc metadata giúp khi câu hỏi nói rõ phạm vi. Câu q23 (bẫy phiên bản) vẫn trượt ở mọi cấu hình, đó là việc tiếp theo."
