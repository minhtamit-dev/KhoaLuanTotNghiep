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

Dự án triển khai và so sánh hiệu năng giữa các mô hình học sâu đơn lẻ (**EfficientNet-B4**, **ViT-B/16**, **ResNet50**) và các phương pháp kết hợp mô hình (**Fixed Soft/Hard Voting**, **Adaptive Ensemble dựa trên Confidence & Entropy**) trên bộ dữ liệu **Chest X-Ray Pneumonia (Kaggle)** với đầy đủ giao diện Web chẩn đoán trực quan và tự động xuất Báo cáo PDF.

---

## 📊 Bảng Kết quả Thực nghiệm (Test Set 624 Ảnh)

| Mô hình / Phương pháp | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B4** | 85.90% | 83.26% | 96.92% | 67.52% | 0.8957 | 0.9401 |
| **ViT-B/16** | 88.14% | 84.20% | **99.74%** | 68.80% | 0.9131 | **0.9799** |
| **ResNet50** | **91.99%** | **89.91%** | 98.21% | **81.62%** | **0.9387** | 0.9708 |
| **Fixed Soft Voting (EffNet + ViT)** | 87.66% | 83.95% | 99.23% | 68.38% | 0.9095 | 0.9710 |
| **Fixed Hard Voting (EffNet + ViT)** | 90.22% | 88.34% | 97.18% | 78.63% | 0.9255 | 0.8791 |
| **Adaptive Ensemble (Confidence)** | 87.66% | 83.95% | 99.23% | 68.38% | 0.9095 | 0.9713 |

> 📌 **Nhận xét:**  
> - **ViT-B/16** đạt Độ nhạy (Recall) cao nhất (**99.74%**), chỉ bỏ sót 1 ca bệnh thực tế trong 390 ca.  
> - **ResNet50** đạt Độ chính xác (Accuracy) cao nhất (**91.99%**) và F1-Score (**0.9387**).  
> - Mô hình Ensemble giúp cân bằng hiệu quả giữa Độ nhạy và Độ đặc hiệu.

---

## 🌐 1. Hướng dẫn Mở Giao diện Web (Streamlit Web UI)

Ứng dụng Web hỗ trợ kéo thả ảnh X-quang, chẩn đoán bằng mô hình Ensemble và hiển thị bản đồ nhiệt **Grad-CAM** giải thích vùng tổn thương.

```bash
# 1. Di chuyển vào thư mục dự án
cd /home/bonkerzz/Documents/doantn/KhoaLuanTotNghiep/pneumonia-adaptive-ensemble

# 2. Kích hoạt venv
source /home/bonkerzz/Documents/doantn/venv/bin/activate

# 3. Mở ứng dụng Web
streamlit run app.py
```
👉 Mở trình duyệt truy cập: **`http://localhost:8501`**

---

## 🚀 2. Hướng dẫn Chạy Thực nghiệm 1-Click (Automated Pipeline)

Tự động chạy toàn bộ quy trình từ tải dữ liệu $\rightarrow$ huấn luyện $\rightarrow$ đánh giá $\rightarrow$ vẽ Grad-CAM $\rightarrow$ biên dịch Báo cáo PDF:

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

# Bước 4: Tạo bản đồ nhiệt Grad-CAM
python scripts/generate_heatmaps.py

# Bước 5: Biên dịch Báo cáo PDF nộp Giảng viên
python scripts/generate_pdf_report.py
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
├── app.py                            # 🌐 Giao diện Web Streamlit minh họa đồ án
├── run_all.py                        # 🚀 Entrypoint 1-click chạy tự động toàn bộ thực nghiệm
├── README.md                         # 📖 Hướng dẫn chi tiết & bảng kết quả
├── requirements.txt                  # 📦 Thư viện phụ thuộc Python
├── CNTT-KLCN171_NguyenGiaKhang.pdf   # 📄 Báo cáo thực nghiệm định dạng PDF (Nhóm trưởng Nguyễn Gia Khang)
├── CNTT-KLCN171_NguyenGiaKhang.docx  # 📄 Báo cáo thực nghiệm gốc định dạng Word
├── thuc_nghiem_tong_ket.md          # 📝 Tổng kết chi tiết thực nghiệm (Markdown)
├── scripts/                          # 🛠️ Tập hợp các script thực thi độc lập
│   ├── prepare_dataset.py            # Tải & liên kết bộ dữ liệu Kaggle
│   ├── train_models.py               # Huấn luyện PyTorch GPU (EfficientNet, ViT, ResNet50)
│   ├── evaluate_models.py            # Đánh giá các mô hình & Ensembles
│   ├── generate_heatmaps.py          # Trực quan hóa Grad-CAM
│   └── generate_pdf_report.py        # Xuất file báo cáo PDF
├── src/                              # 🧠 Mã nguồn lõi Python
│   ├── data/dataset.py               # PyTorch Dataset wrapper
│   ├── models/factory.py             # Model Factory linh hoạt
│   ├── evaluation/metrics.py         # Hàm tính toán 7 độ đo y khoa
│   ├── ensemble/                     # Module Soft/Hard Voting & Adaptive Weighting
│   └── explainability/gradcam.py     # Thuật toán Grad-CAM thuần PyTorch
├── models/                           # 💾 Trọng số mô hình (.pt)
└── results/                          # 📊 Kết quả CSV, Biểu đồ ROC, Confusion Matrix & Grad-CAM
```