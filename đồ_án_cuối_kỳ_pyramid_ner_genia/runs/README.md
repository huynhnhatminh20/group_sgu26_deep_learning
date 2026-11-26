# Hướng dẫn thư mục `runs/`

Thư mục này chứa các module runtime của package Kaggle cùng các công cụ local. Module runtime giữ nguyên đường dẫn đã nghiệm thu; công cụ local được nhóm vào `configs/`, `diagnostics/`, `tests/`. Local tools use canonical module paths from repository root.

## Đường chạy chính thức

| Kiến trúc | Entrypoint huấn luyện | Cấu hình |
|---|---|---|
| Ju B0 | `python -m baselines.ju2018.train` | `runs/full_matrix.py` tạo command, GENIA `protocol_clean_v1`, BioWordVec frozen |
| Pyramid Basic/Full, O/H/A/T | `python train_ner.py` | `runs/full_matrix.py` tạo command và 31 manifests; `runs/package_full_v1.py` tạo notebook Kaggle theo run ID |

Checkpoint được chọn bằng VALID; TEST chỉ dành cho final evaluation. Hai run lịch sử B0-42 và R0-42 đã hoàn thành, không chạy đè. Package source cuối đã nghiệm thu là `pyramid_full_v1_3435844c649b.zip` (SHA-256 `a3fcb2ed6f8aa195aadfcb9d4b7390fe2e075ec95b40c2f66ee7679f1c9f1e0f`) ở ngoài repository. Việc chạy lại builder sẽ tạo package mới nếu source thay đổi; tài liệu và diagnostic notebook không nằm trong đường training chính thức.

## Vai trò từng file

| File | Vai trò | Được dùng bởi |
|---|---|---|
| `full_matrix.py` | Định nghĩa 31 run, hyperparameters, command và kiểm tra cấu hình đã khóa. Không tự huấn luyện. | `package_full_v1.py`, `experiment_matrix.py`, preflight, workbook |
| `configs/experiment_matrix.py` | In/kiểm tra danh mục 31 run. | python -m runs.configs.experiment_matrix |
| `package_full_v1.py` | Tạo ZIP source có manifest/SHA và 31 notebook Kaggle. | Bước packaging, không nằm trong training runtime |
| `checkpointing.py` | Lưu/nạp trạng thái model, optimizer, RNG và metadata tại ranh giới epoch. | B0 và trainer Pyramid |
| `network_debug.py` | Monitor scalar tùy chọn: gradients, activations, weights, updates, NaN/Inf. Mặc định OFF. | B0/Pyramid khi `NETWORK_DEBUG=1`; notebook debug |
| `wandb_auth.py` | Đọc credential từ nguồn riêng ngoài Git. | `wandb_tracking.py` |
| `wandb_tracking.py` | Tracking tùy chọn, ghi `tracking_state.json`, run ID, stop/failure state. | B0/Pyramid; `excel/update_workbook.py` đọc artifact |
| `cpu_preflight_v1.py` | Kiểm tra một mini-batch TRAIN trên CPU, checkpoint và input hashes; không optimizer step/TEST F1. | Notebook Kaggle preflight |
| `validate_contextual_v1.py` | Kiểm tra alignment, chiều và checksum contextual pickle trên TRAIN/VALID. | H1-H6 preflight |
| `gen_bert_emb.py` | Generator BioBERT của source tác giả. | H1 |
| `gen_albert_emb.py` | Generator ALBERT của source tác giả. | H2 |
| `gen_bert_flair_emb.py` | Generator BioBERT + Flair của source tác giả. | H3; H6 dùng lại artifact H3 |
| `gen_albert_flair_emb.py` | Generator ALBERT + Flair của source tác giả. | H4 |
| `gen_albert_bert_emb.py` | Generator BioBERT + ALBERT của source tác giả. | H5 |
| `reduce_wv.py` | Utility gốc xử lý word vectors; không phải training entrypoint cho 31 run hiện tại. | Nguồn tác giả, giữ để truy xuất |
| `measure_throughput_v1.py` | Đo throughput inference trên VALID, phục vụ phân tích chi phí; không đọc TEST. | Diagnostic RQ2 |
| `diagnostics/debug_small_overfit.py` | Diagnostic CPU cô lập trên hai câu TRAIN; ghi loss/prediction/gradients theo bước. Không ghi đè full runs. | python -m runs.diagnostics.debug_small_overfit |
| `tests/test_full_configs.py` | Regression cấu hình B0/Pyramid Basic/Full và A1-A5. | python -m runs.tests.test_full_configs |
| `tests/test_genia_dataloader.py` | Regression GENIA collation, labels và masks. | python -m runs.tests.test_genia_dataloader |
| `tests/test_leakage_guards.py` | Regression TRAIN-only vocabulary, TEST isolation, BioWordVec và checkpoint. | python -m runs.tests.test_leakage_guards |
| `tests/test_network_debug.py` | Kiểm tra monitor ON/OFF không đổi optimizer step, RNG, logits/loss/gradients và phát hiện NaN. | python -m runs.tests.test_network_debug |
| `../notebooks/Kaggle_Wandb_Diagnostic.ipynb` | Notebook kiểm tra W&B/Secret bằng dữ liệu synthetic; không phải research run. | Chẩn đoán riêng, không dùng để báo F1 |

## Đọc kết quả và cập nhật Excel

- `notebooks/Network_Debugging_GENIA_V1.ipynb` giải thích các kiểm tra CPU và hiển thị output thực; phần overfit đọc `excel/debug_diagnostics/`.
- `excel/update_workbook.py` đọc 31 cấu hình, history/checkpoint, `tracking_state.json`, debug JSONL và preflight evidence để tạo `excel/GENIA_Experiments.xlsx`. Script này chạy offline, không được import trong training.
- Hai nguồn kết quả lịch sử nằm dưới `../Pyramid_Kaggle_Pilot_V1_Upload/`; các output tương lai theo `runs/results/<ID>/seed<seed>/`. `Run_Log` chỉ có hàng cho run đã quan sát, không tạo lỗi/metric giả cho run chưa chạy.
