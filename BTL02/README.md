# BTL02 — Nhận dạng cảm xúc khuôn mặt trên FER2013 bằng CNN

**Nhóm SGU26 — Chuyên đề Học sâu**
Sinh viên: Huỳnh Nhật Minh — 3123580025

## Thành phần bài nộp

| Tệp / Thư mục | Nội dung |
|---|---|
| `BTL02-Huỳnh_Nhật_Minh_3123580025.docx` | Báo cáo chi tiết bài tập lớn dạng Word |
| `BTL02-Huỳnh_Nhật_Minh_3123580025.pdf` | Báo cáo chi tiết bài tập lớn dạng PDF |
| `slide.pptx` | Slide thuyết trình bài tập lớn dạng PowerPoint |
| `slide.pdf` | Slide thuyết trình bài tập lớn dạng PDF |
| `btl02_fer2013.ipynb` | Toàn bộ mã nguồn PyTorch: nạp dữ liệu, EDA, augmentation, định nghĩa mô hình, 27 lượt thực nghiệm nhóm A/B/C, mô hình cuối, đánh giá và trực quan bộ lọc |
| `outputs/experiment_log.xlsx` | Nhật ký thực nghiệm 7 sheet: tổng quan, chi tiết 28 lượt, tổng hợp A-B-C, cấu hình, lịch sử chạy, log từng epoch |
| `outputs/` | Chứa biểu đồ `figures/`, log từng epoch `logs/`, các bảng tổng hợp `.csv` và kết quả đánh giá |
| `đề BTL02/` | Đề bài yêu cầu (`Emotion_Recognition.pdf`), báo cáo mẫu và mã nguồn demo |
| `requirements.txt` | Danh sách thư viện Python cần cài đặt |

## Cấu trúc thư mục

```text
BTL02/
├── BTL02-Huỳnh_Nhật_Minh_3123580025.docx   Báo cáo Word
├── BTL02-Huỳnh_Nhật_Minh_3123580025.pdf    Báo cáo PDF
├── slide.pptx                             Slide thuyết trình PowerPoint
├── slide.pdf                              Slide thuyết trình PDF
├── btl02_fer2013.ipynb                    Notebook chính huấn luyện và đánh giá
├── requirements.txt                       Thư viện cần cài
├── README.md                              Tài liệu hướng dẫn và mô tả
├── đề BTL02/                              Đề bài và tài liệu tham khảo demo
├── data/                                  Thư mục dữ liệu chứa fer2013.csv (git ignored)
└── outputs/
    ├── figures/                           Biểu đồ trực quan hóa (EDA, hội tụ, confusion matrix, bộ lọc...)
    ├── logs/                              28 file log chi tiết các epoch
    ├── architecture_table.csv
    ├── attempts_log.csv
    ├── classification_report.csv
    ├── class_distribution.csv
    ├── config.json
    ├── experiment_log.xlsx
    ├── summary_groups.csv
    ├── summary_runs.csv
    └── test_predictions.csv
```

## Chạy notebook

1. Cài Python 3.11 và thư viện: `python -m pip install -r requirements.txt`.
2. Tải dữ liệu (cần đăng nhập Kaggle CLI):
   `kaggle datasets download -d deadskull7/fer2013 -p data --unzip`
   hoặc đặt thủ công `fer2013.csv` vào `data/`.
3. Mở `btl02_fer2013.ipynb` và chọn **Restart Kernel and Run All Cells**.

Notebook tự cache dữ liệu vào `outputs/data/fer2013.npz`. Trên GPU 4 GB, toàn bộ thực nghiệm mất khoảng 2,5 giờ. Có thể giảm `SEEDS` hoặc `epochs` trong ô cấu hình để chạy nhanh; mọi bảng/hình sẽ được sinh lại.

## Thực nghiệm

- **Nhóm A — kiến trúc:** Small (2-2-2 conv, 32-64-128), Medium (3-3-3, 64-128-256), Large (2-2-2-2, 64-128-256-512).
- **Nhóm B — kỹ thuật huấn luyện:** augmentation, BatchNorm, augmentation + BatchNorm; baseline là A2.
- **Nhóm C — tối ưu:** SGD momentum 0.9, LeakyReLU, Adam lr = 5e-4; baseline là A2.
- Mỗi cấu hình chạy 3 seed (11, 22, 33), 20 epoch, early stopping theo validation.
- Mô hình cuối chọn theo validation, huấn luyện lại 45 epoch với cosine decay, chỉ đánh giá PrivateTest một lần.

## Kết quả

- **Nhóm A (test accuracy):** Small 60,13% — Medium 60,31% — Large 61,46%.
- **Nhóm B (test accuracy):** baseline 60,31% — augmentation 62,46% — BatchNorm 51,73% — augmentation + BatchNorm 48,67%.
- **Nhóm C (test accuracy):** SGD 58,77% — LeakyReLU 49,45% (một seed phân kỳ) — Adam 5e-4 61,60%.
- **Mô hình cuối:** Medium + augmentation, 45 epoch, cosine decay — validation 64,95%, test **65,67%**, macro-F1 **61,66%** trên PrivateTest.

Xem `outputs/summary_groups.csv`, `outputs/summary_runs.csv`, `outputs/experiment_log.xlsx`, `outputs/attempts_log.csv` và các hình trong `outputs/figures/`. Báo cáo và slide lấy trực tiếp số liệu từ các tệp này.

