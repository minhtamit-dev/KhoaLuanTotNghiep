# 🫁 Adaptive CNN–ViT Ensemble for Pneumonia Detection

> **Đề tài:** Nghiên cứu phương pháp ensemble cố định và thích ứng giữa CNN và Vision Transformer trong nhận diện viêm phổi từ ảnh X-quang ngực.  
> **Mã đề tài:** CNTT-KLCN171  
> **Giảng viên hướng dẫn:** TS. Phùng Thế Bảo  
> **Nhóm thực hiện:**  
> 1. **Nguyễn Gia Khang** (Nhóm trưởng - MSSV: 2001230377)  
> 2. **Phan Minh Tâm** (MSSV: 2001230785)  
> 3. **Trịnh Minh Hiếu** (MSSV: 2001230245)  

---

## 📋 Tổng quan Dự án

Dự án triển khai và so sánh hiệu năng giữa các mô hình học sâu đơn lẻ (**EfficientNet-B4**, **ViT-B/16**, **ResNet50**) và các phương pháp kết hợp mô hình (**Fixed Soft/Hard Voting**, **Weighted Voting**, **Adaptive Ensemble**) trên bộ dữ liệu **Chest X-Ray Pneumonia (Kaggle)** với đầy đủ giao diện Web chẩn đoán trực quan và giải thích bằng **Grad-CAM**.

---

## 📊 Bảng Kết quả Thực nghiệm (Test Set 624 Ảnh)

| Mô hình / Phương pháp | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B4** | 85.58% | 81.78% | 98.97% | 63.25% | 0.8956 | 0.9654 |
| **ViT-B/16** | 84.46% | 80.33% | 99.49% | 59.40% | 0.8889 | **0.9722** |
| **ResNet50** | 84.29% | 80.04% | **99.74%** | 58.55% | 0.8881 | 0.9652 |
| **Fixed Soft Voting (EffNet + ViT)** | 85.10% | 81.00% | 99.49% | 61.11% | 0.8930 | 0.9700 |
| **Ensemble (ResNet50 + ViT)** | 84.62% | 80.37% | **99.74%** | 59.40% | 0.8902 | 0.9663 |
| **Tri-Ensemble (EffNet + ViT + ResNet)** | **85.42%** | **81.21%** | **99.74%** | **61.54%** | **0.8953** | 0.9708 |

> 📌 **Tại ngưỡng tối ưu Youden (Validation th=0.61):** `Fixed Soft Voting (EffNet + ViT)` đạt **Accuracy 87.34%**, **F1-Score 0.9074** và **Specificity 67.52%**, vượt trội so với các mô hình đơn lẻ.

---

## 🌐 1. Hướng dẫn Mở Giao diện Web (Streamlit Web UI)

Ứng dụng Web hỗ trợ kéo thả ảnh X-quang, chẩn đoán bằng mô hình Ensemble và hiển thị bản đồ nhiệt **Grad-CAM & ViT Attention** giải thích vùng tổn thương.

```bash
# 1. Kích hoạt môi trường ảo (nếu có)
# Windows: .\.venv\Scripts\Activate.ps1
# Linux:   source .venv/bin/activate

# 2. Mở ứng dụng Web
streamlit run app.py
```
👉 Mở trình duyệt truy cập: **`http://localhost:8501`**

---

## 🚀 2. Hướng dẫn Chạy Thực nghiệm 1-Click (Automated Pipeline)

Tự động chạy toàn bộ quy trình từ tải dữ liệu $\rightarrow$ huấn luyện $\rightarrow$ đánh giá $\rightarrow$ vẽ Grad-CAM:

```bash
python run_all.py
```

---

## 🛠️ 3. Hướng dẫn Chạy Từng Bước Thủ công (Manual Execution)

```bash
# Bước 1: Chuẩn bị & liên kết dữ liệu Kaggle Chest X-Ray
python scripts/prepare_dataset.py

# Bước 2: Huấn luyện các mô hình trên PyTorch GPU
python scripts/train_models.py

# Bước 3: Đánh giá & tính toán chỉ số (Xuất CSV + Biểu đồ ROC)
python scripts/evaluate_models.py

# Bước 4: Chạy kiểm thử 10 lần chạy & thống kê độ chính xác
python scripts/run_10_evaluations.py

# Bước 5: Tạo bản đồ nhiệt Grad-CAM & Attention
python scripts/generate_heatmaps.py
```

---

## 🔄 4. Hướng dẫn Thay đổi / Mở rộng Mô hình Thực nghiệm

Hệ thống được thiết kế theo kiến trúc **Modular Factory**, cho phép bạn thay đổi hoặc thêm bất kỳ kiến trúc mô hình mới nào từ thư viện `timm` (*DenseNet121, ConvNeXt, Swin Transformer...*):

1. **Sửa danh sách mô hình muốn huấn luyện** trong `scripts/train_models.py`:
   ```python
   models_to_train = ['efficientnet_b4', 'vit_b16', 'densenet121', 'convnext_base']
   for name in models_to_train:
       train_model(name, epochs=3)
   ```
2. **Khai báo alias** trong `src/models/factory.py`:
   ```python
   MODEL_ALIAS = {'densenet121': 'densenet121', 'convnext_base': 'convnext_base'}
   ```
3. **Đánh giá lại**: `python scripts/evaluate_models.py`

---

## 📁 Cấu trúc Thư mục Dự án

```text
pneumonia-adaptive-ensemble/
├── app.py                            # 🌐 Giao diện Web Streamlit chẩn đoán & giải thích
├── run_all.py                        # 🚀 Entrypoint 1-click chạy tự động toàn bộ thực nghiệm
├── README.md                         # 📖 Hướng dẫn chi tiết & bảng kết quả
├── requirements.txt                  # 📦 Thư viện phụ thuộc Python
├── scripts/                          # 🛠️ Tập hợp các script thực thi độc lập
│   ├── prepare_dataset.py            # Tải & liên kết bộ dữ liệu Kaggle
│   ├── train_models.py               # Huấn luyện PyTorch GPU (EfficientNet, ViT, ResNet50)
│   ├── evaluate_models.py            # Đánh giá các mô hình & Ensembles
│   ├── run_10_evaluations.py         # Kiểm thử 10 lần chạy & thống kê độ chính xác
│   └── generate_heatmaps.py          # Trực quan hóa Grad-CAM & Attention Maps
├── src/                              # 🧠 Mã nguồn lõi Python
│   ├── data/dataset.py               # PyTorch Dataset wrapper (Patient-aware split)
│   ├── models/factory.py             # Model Factory linh hoạt
│   ├── evaluation/metrics.py         # Hàm tính toán độ đo y khoa & Youden threshold
│   ├── ensemble/                     # Module Soft/Hard Voting & Adaptive Weighting
│   ├── explainability/gradcam.py     # Thuật toán Grad-CAM và ViT Attention
│   └── utils/database.py             # SQLite cơ sở dữ liệu y tế & khóa ngoại
├── models/                           # 💾 Trọng số mô hình (.pt)
└── results/                          # 📊 Kết quả CSV, Biểu đồ ROC, Confusion Matrix & Grad-CAM
```