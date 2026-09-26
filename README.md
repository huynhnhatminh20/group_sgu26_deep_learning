# Group SGU26 - Deep Learning

Repository dành cho bài tập và dự án môn **Deep Learning** của nhóm SGU26.

---

## 📌 Hướng dẫn làm việc với Git cho nhóm (Git Workflow)

### 1. Khởi tạo & Clone Repository về máy
Nếu chưa tải repo về máy, mở Terminal / PowerShell và chạy:
```bash
git clone https://github.com/huynhnhatminh20/group_sgu26_deep_learning.git
cd group_sgu26_deep_learning
```

---

### 2. Quy trình làm việc hàng ngày (Daily Workflow)

#### Bước 1: Cập nhật code mới nhất từ nhánh `main`
Trước khi bắt đầu làm bài hoặc code tính năng mới, luôn đồng bộ nhánh chính:
```bash
git checkout main
git pull origin main
```

#### Bước 2: Tạo nhánh riêng để làm việc
Tránh commit trực tiếp lên nhánh `main`. Hãy tạo nhánh mới cho tính năng/bài tập của bạn:
```bash
# Cú pháp: git checkout -b feat/<ten-tinh-nang-hoac-ten-thanh-vien>
git checkout -b feat/tuan-1-data-preprocessing
```

#### Bước 3: Thêm file và Commit thay đổi
Sau khi viết code / làm notebook:
```bash
# Xem trạng thái các file đã sửa
git status

# Thêm các file thay đổi vào staging
git add .

# Commit kèm thông điệp rõ ràng
git commit -m "feat: hoàn thành tiền xử lý dữ liệu bài tập tuần 1"
```

#### Bước 4: Đẩy nhánh lên GitHub
```bash
git push -u origin feat/tuan-1-data-preprocessing
```

#### Bước 5: Tạo Pull Request (PR)
1. Truy cập vào GitHub: [group_sgu26_deep_learning](https://github.com/huynhnhatminh20/group_sgu26_deep_learning)
2. Bấm **Compare & pull request**.
3. Gửi cho các thành viên khác trong nhóm review và tiến hành merge vào nhánh `main`.

---

### 3. Quy ước đặt tên Commit (Conventional Commits)

| Tiền tố | Mục đích | Ví dụ |
|---|---|---|
| `feat:` | Thêm tính năng mới / bài lab mới | `feat: thêm notebook cnn_cifar10` |
| `fix:` | Sửa lỗi trong code / mô hình | `fix: sửa lỗi thiếu import tensorflow` |
| `docs:` | Cập nhật tài liệu / README | `docs: cập nhật hướng dẫn git` |
| `refactor:` | Tối ưu hoặc dọn dẹp lại code | `refactor: tái cấu trúc module dataloader` |
| `chore:` | Cập nhật config, requirements, gitignore | `chore: thêm requirements.txt` |

---

### 4. Tổng hợp các lệnh Git thường dùng

```bash
# Kiểm tra trạng thái
git status

# Xem lịch sử commit
git log --oneline --graph --all

# Chuyển nhánh
git checkout <ten-nhanh>

# Hủy bỏ thay đổi của một file chưa stage
git checkout -- <duong-dan-file>

# Lấy code mới nhất về nhưng chưa gộp
git fetch origin
```
