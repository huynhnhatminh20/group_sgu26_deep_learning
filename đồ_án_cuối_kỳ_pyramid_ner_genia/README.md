# Pyramid trên văn bản y sinh GENIA — Nested Named Entity Recognition

Học phần Học sâu
Nhóm 9 gồm 1 thành viên
Họ tên sinh viên Huỳnh Nhật Minh
Lớp DDU1231
Mã số sinh viên 3123580025


Đồ án này áp dụng kiến trúc Pyramid cho bài toán Nhận dạng Thực thể Tên Lồng nhau (Nested NER) trên tập dữ liệu y sinh GENIA.

## Mục tiêu nghiên cứu
- **RQ1:** Đánh giá và so sánh mô hình Pyramid-Full với baseline đối chứng Layered BiLSTM–CRF (Ju et al., 2018).
- **RQ2:** Khảo sát ảnh hưởng của cơ chế Inverse Pyramid cùng độ sâu kim tự tháp $L$.

## Cấu trúc dự kiến
- `data/`, `utils/`: Module nạp và tiền xử lý dữ liệu GENIA.
- `layers/`, `functions/`: Các tầng mạng cốt lõi (CRF, Encodings, Routing).
- `models/`: Định nghĩa kiến trúc Pyramid và Flat tagger.
- `baselines/`: Mô hình đối chứng Ju et al. (2018).
- `notebooks/`: Nhật ký thực nghiệm và các pipeline huấn luyện.
