# examples_pytorch

Cấu trúc thư mục được sắp xếp giống hệt gói `examples.zip` gốc, toàn bộ 12 notebook đã chuyển đổi sang **PyTorch**:

```text
examples_pytorch/
├── 0. Getting Started/
│   ├── getting_started_pytorch.ipynb
│   └── Iris_Bayesian_decision_making_pytorch.ipynb
├── 1. MLP/
│   ├── Func_apxm_MLP_pytorch.ipynb
│   └── MNIST_MLP_pytorch.ipynb
├── 2. CNN + MLP/
│   ├── 1_MNIST_CNN_pytorch.ipynb
│   ├── 2_Human_Gender_CNN_pytorch.ipynb
│   └── 3_Emotion_Reg_CNN_pytorch.ipynb
├── 3. CNN + LSTM/
│   └── MNIST_CNN_LSTM_pytorch.ipynb
├── 4. Training Utilities/
│   ├── 2_Human_Gender_CNN_pytorch.ipynb   (Checkpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard)
│   └── 3_Emotion_Reg_CNN_pytorch.ipynb    (Data Augmentation + EarlyStopping)
├── 5. Others/
│   ├── 1_segnet_pytorch.ipynb
│   └── 2_obj_detection_pytorch.ipynb
├
├── requirements.txt
└── README_PyTorch.md
```

## Cài đặt thư viện

```bash
pip install -r requirements.txt
```


## Cấu trúc dữ liệu

```text
data/
├── gender/          # 4 file .npy Human Gender + data_human_gender.zip
├── fer2013/         # fer2013_training_onehot.csv, fer2013_publictest_onehot.csv
├── obj_detection/   # vehicles/, non-vehicles/, test.jpg
├── segnet/          # train/, test/
└── MNIST/           # tự động tải qua torchvision
```

Chạy notebook từ thư mục gốc `BTL03` để các đường dẫn tương đối ở trên hoạt động.
