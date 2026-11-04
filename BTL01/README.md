# BTL01 — Xấp xỉ Gaussian bằng MLP

## Chạy bằng notebook

Chỉ sử dụng `btl01.ipynb`; không cần file Python riêng.

1. Cài Python 3.10 trở lên và các thư viện: `python -m pip install -r requirements.txt`.
2. Mở `btl01.ipynb` trong Jupyter hoặc VS Code và chọn kernel đã cài thư viện.
3. Giữ `GROUP = "all"`, chọn **Restart Kernel and Run All Cells** để chạy đủ 24 cấu hình × 3 seed = 72 lượt.
4. Để chạy một nhóm, đổi `GROUP` ở cell đường dẫn thành `"A"`, `"B"` hoặc `"C"`, rồi chạy lại từ đầu. Kết quả riêng được lưu dưới `outputs/group_A/`, `group_B/` hoặc `group_C/`.

## Cấu hình

- Gaussian: μ = [0, 0], C = [[0.3, -0.5], [-0.5, 2]].
- Miền lấy đều: x1 ∈ [-2, 2], x2 ∈ [-4, 4].
- Huấn luyện 2.000 mẫu; validation 256; test độc lập 100. Seed dữ liệu 2026, 2027, 2028.
- A: 200–2.000 mẫu, bước 200; một tầng ẩn, 10 neuron.
- B: 1–10 neuron; một tầng ẩn, 2.000 mẫu.
- C: 1–4 tầng ẩn; 10 neuron/tầng, 2.000 mẫu.
- Ba seed mô hình: 11, 22, 33. Kết quả báo cáo dạng trung bình ± độ lệch chuẩn.
- Adam, learning rate 0.003, batch 128, chạy cố định 100 epoch; không dừng sớm.
- Tanh trong tầng ẩn; Sigmoid ở đầu ra nhân với mật độ cực đại. Chuẩn hóa tọa độ và mật độ khi huấn luyện; các MSE báo cáo được đổi về thang gốc.
- Checkpoint tốt nhất theo validation; chọn cấu hình trước khi đánh giá test. Chạy trên CPU để code dễ đọc và dễ tái lập.

## Cấu trúc thư mục

```text
BTL01/
├── btl01.ipynb        notebook chính, chạy toàn bộ thực nghiệm
├── requirements.txt   thư viện cần cài
├── README.md          hướng dẫn chạy và mô tả thư mục
├── báo cáo.docx       báo cáo bản Word
├── báo cáo.pdf        báo cáo bản PDF
├── slides.tex         slide Beamer
├── slides.pdf         slide PDF 15 trang
├── figures/           9 hình dùng cho báo cáo và slide
└── outputs/
    ├── data/          train.csv, validation.csv, test.csv
    ├── logs/          72 file loss theo epoch
    ├── models/        72 checkpoint .pt
    ├── figures/       9 hình PNG do notebook sinh ra
    ├── config.json
    ├── summary_runs.csv
    ├── summary_abc.csv
    ├── selected_model.json
    └── test_predictions.csv
```

## Kết quả

- `outputs/data/`: ba tập CSV.
- `outputs/logs/`: loss huấn luyện và validation theo epoch của 72 lần chạy.
- `outputs/summary_runs.csv`: kết quả từng seed.
- `outputs/summary_abc.csv`: bảng tổng hợp A/B/C dạng trung bình ± độ lệch chuẩn.
- `outputs/models/`: 72 checkpoint để kiểm chứng dự đoán.
- `outputs/figures/`: dữ liệu, phân bố, sai số A/B/C, hội tụ, thời gian và mô hình cuối.
- `outputs/selected_model.json`: cấu hình, seed và chỉ số mô hình được chọn.
- `outputs/test_predictions.csv`: 100 dự đoán của mô hình cuối.
- `outputs/config.json`: cấu hình chung, seed và kích thước ba tập dữ liệu.

Notebook dùng biến cấu hình dạng từ điển, một vòng lặp cho 24 cấu hình × 3 seed và các cell tuần tự. Các hàm riêng chỉ dùng cho phần được gọi lại như sinh dữ liệu, huấn luyện, dự đoán và lưu hình.
