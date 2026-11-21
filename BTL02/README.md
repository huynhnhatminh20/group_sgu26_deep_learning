# BTL02 — Nhận dạng cảm xúc khuôn mặt trên FER2013 bằng CNN

**Nhóm SGU26 — Chuyên đề Học sâu**
Sinh viên: Huỳnh Nhật Minh — 3123580025

## Thành phần bài nộp

| Tệp | Nội dung |
|---|---|
| `btl02_fer2013.ipynb` | Toàn bộ mã nguồn PyTorch: nạp dữ liệu, EDA, augmentation, định nghĩa mô hình, 27 lượt thực nghiệm nhóm A/B/C, mô hình cuối, đánh giá và trực quan bộ lọc |
| `report.tex` | Báo cáo LaTeX (XeLaTeX), không kèm PDF |
| `slides.tex` | Slide Beamer LaTeX (XeLaTeX) |
| `outputs/experiment_log.xlsx` | Nhật ký thực nghiệm 7 sheet: tổng quan, chi tiết 28 lượt, tổng hợp A-B-C, cấu hình, lịch sử chạy (kể cả FAILED/DEBUG), lịch sử phát triển, log từng epoch |
| `outputs/` | Log từng epoch, `attempts_log.csv`, hình, bảng tổng hợp, checkpoint mô hình cuối |
| `requirements.txt` | Thư viện Python cần cài |

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

## Biên dịch LaTeX

Cả hai tệp dùng XeLaTeX (do có tiếng Việt và font hệ thống):

```bash
xelatex report.tex
xelatex slides.tex
```
