# Adaptive CNN–ViT Ensemble for Pneumonia Detection

> Nghiên cứu phương pháp ensemble thích ứng giữa CNN và Vision Transformer trong nhận diện viêm phổi từ ảnh X-quang ngực.

---

## 1. Giới thiệu đề tài

Đề tài tập trung nghiên cứu phương pháp kết hợp giữa hai hướng kiến trúc Deep Learning phổ biến trong bài toán phân loại ảnh y tế:

- **EfficientNet-B4** — đại diện cho kiến trúc CNN.
- **Vision Transformer ViT-B/16** — đại diện cho kiến trúc Transformer.

Mục tiêu là xây dựng và đánh giá một phương pháp **ensemble thích ứng (Adaptive Ensemble)** nhằm tận dụng ưu điểm của cả hai mô hình trong bài toán nhận diện viêm phổi từ ảnh X-quang ngực.

Thay vì chỉ sử dụng một mô hình duy nhất, đề tài tiến hành thực nghiệm theo các mức:

1. EfficientNet-B4 làm mô hình baseline.
2. ViT-B/16 làm mô hình baseline.
3. Fixed Ensemble giữa EfficientNet-B4 và ViT-B/16.
4. Adaptive Ensemble dựa trên mức độ tin cậy hoặc độ bất định của từng mô hình.
5. So sánh toàn diện các phương pháp.
6. Sử dụng phương pháp Explainable AI để trực quan hóa vùng ảnh ảnh hưởng đến quyết định.
7. Tích hợp mô hình tốt nhất vào ứng dụng web minh họa.

---

# 2. Mục tiêu

## 2.1. Mục tiêu tổng quát

Xây dựng và đánh giá phương pháp ensemble thích ứng giữa EfficientNet-B4 và Vision Transformer ViT-B/16 cho bài toán nhận diện viêm phổi từ ảnh X-quang ngực.

---

## 2.2. Mục tiêu cụ thể

### Mục tiêu 1 — Khảo sát dữ liệu

- Khảo sát bộ dữ liệu X-quang ngực.
- Phân tích phân bố các lớp.
- Kiểm tra chất lượng ảnh.
- Xác định các vấn đề liên quan đến dữ liệu.
- Xây dựng quy trình preprocessing và augmentation phù hợp.

### Mục tiêu 2 — Xây dựng CNN baseline

Huấn luyện và đánh giá EfficientNet-B4 trên cùng một bộ dữ liệu.

### Mục tiêu 3 — Xây dựng Transformer baseline

Huấn luyện và đánh giá ViT-B/16 trên cùng một bộ dữ liệu.

### Mục tiêu 4 — Xây dựng Fixed Ensemble

Kết hợp dự đoán của EfficientNet-B4 và ViT-B/16 bằng phương pháp ensemble cố định.

Ví dụ:

    P_final = α × P_CNN + (1 - α) × P_ViT

Trong đó α là trọng số của EfficientNet-B4.

Giá trị α cần được xác định thông qua thiết kế thực nghiệm phù hợp.

### Mục tiêu 5 — Xây dựng Adaptive Ensemble

Đề xuất cơ chế xác định trọng số của từng mô hình dựa trên thông tin dự đoán của mô hình, chẳng hạn:

- Prediction confidence.
- Prediction uncertainty.
- Hoặc các phương pháp định lượng độ tin cậy khác được lựa chọn trong quá trình nghiên cứu.

Mục tiêu là để trọng số của EfficientNet-B4 và ViT-B/16 có thể thay đổi theo từng ảnh thay vì sử dụng một trọng số cố định cho toàn bộ dataset.

### Mục tiêu 6 — Đánh giá

So sánh:

- EfficientNet-B4.
- ViT-B/16.
- Fixed Ensemble.
- Adaptive Ensemble.

Thông qua các metric:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Sensitivity.
- Specificity.

Ngoài ra có thể đánh giá thêm khả năng calibration và độ tin cậy của xác suất dự đoán nếu phù hợp với phạm vi khóa luận.

### Mục tiêu 7 — Explainability

Sử dụng các phương pháp Explainable AI để trực quan hóa vùng ảnh có ảnh hưởng đến quyết định của mô hình.

Đối với CNN, sử dụng Grad-CAM hoặc phương pháp tương đương.

Đối với Vision Transformer, nghiên cứu phương pháp visualization phù hợp với kiến trúc Transformer.

### Mục tiêu 8 — Web Demo

Tích hợp mô hình tốt nhất vào một ứng dụng web minh họa.

Ứng dụng cho phép:

1. Upload ảnh X-quang.
2. Tiền xử lý ảnh.
3. Chạy mô hình.
4. Hiển thị kết quả dự đoán.
5. Hiển thị xác suất / độ tin cậy.
6. Hiển thị trọng số ensemble nếu phù hợp.
7. Hiển thị vùng ảnh được mô hình chú ý / nghi ngờ.

> **Lưu ý:** Ứng dụng chỉ nhằm mục đích minh họa nghiên cứu, không phải công cụ chẩn đoán y khoa.

---

# 3. Research Pipeline

Pipeline nghiên cứu tổng thể:

```text
                         CHEST X-RAY DATASET
                                  │
                                  ▼
                         DATA EXPLORATION
                                  │
                                  ▼
                    DATA PREPROCESSING / SPLIT
                                  │
                                  ▼
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
              EfficientNet-B4               ViT-B/16
                    │                           │
                    ▼                           ▼
              CNN Baseline                 ViT Baseline
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
              Fixed Ensemble            Adaptive Ensemble
                    │                           │
                    │                 ┌─────────┴─────────┐
                    │                 │                   │
                    │                 ▼                   ▼
                    │          Confidence-based    Uncertainty-based
                    │
                    └─────────────┬───────────────────────┘
                                  │
                                  ▼
                            EVALUATION
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
             Metrics         Explainability    Error Analysis
                │                 │                 │
                └─────────────────┼─────────────────┘
                                  │
                                  ▼
                         MODEL COMPARISON
                                  │
                                  ▼
                         BEST MODEL SELECTION
                                  │
                                  ▼
                         WEB DEMONSTRATION
```

---
            
# 4. Research Pipeline
Cấu trúc project được thiết kế theo hướng:

- Tách dữ liệu khỏi source code.
- Tách model architecture khỏi training.
- Tách ensemble khỏi model.
- Tách evaluation khỏi training.
- Tách experiments khỏi source code.
- Tách trained weights khỏi source code.
- Tách research pipeline khỏi web application.

Mục tiêu là project vừa phục vụ nghiên cứu khoa học, vừa có thể mở rộng thành một ứng dụng demo.

---

# 5. Project Structure
pneumonia-adaptive-ensemble/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── external/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_efficientnet_baseline.ipynb
│   ├── 04_vit_baseline.ipynb
│   ├── 05_fixed_ensemble.ipynb
│   ├── 06_adaptive_ensemble.ipynb
│   ├── 07_explainability.ipynb
│   └── 08_final_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── ensemble/
│   ├── evaluation/
│   ├── explainability/
│   ├── inference/
│   └── utils/
│
├── configs/
│   ├── dataset.yaml
│   ├── efficientnet_b4.yaml
│   ├── vit_b16.yaml
│   ├── fixed_ensemble.yaml
│   ├── adaptive_ensemble.yaml
│   └── experiment.yaml
│
├── experiments/
│   ├── baseline/
│   │   ├── efficientnet_b4/
│   │   └── vit_b16/
│   ├── fixed_ensemble/
│   ├── adaptive_ensemble/
│   └── ablation/
│
├── models/
│   ├── efficientnet_b4/
│   │   ├── best.pt
│   │   └── config.yaml
│   ├── vit_b16/
│   │   ├── best.pt
│   │   └── config.yaml
│   └── ensemble/
│       └── config.yaml
│
├── results/
│   ├── metrics/
│   ├── plots/
│   ├── roc_curves/
│   ├── confusion_matrices/
│   ├── calibration/
│   ├── heatmaps/
│   └── tables/
│
├── api/
│   ├── routes/
│   │   ├── prediction.py
│   │   └── health.py
│   ├── schemas/
│   │   └── prediction.py
│   ├── services/
│   │   ├── inference_service.py
│   │   └── explainability_service.py
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_efficientnet.py
│   ├── train_vit.py
│   ├── evaluate_models.py
│   └── generate_heatmaps.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_ensemble.py
│   └── test_inference.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env

---

# 6. Giải thích từng thư mục

## 6.1. data/
Chứa toàn bộ dữ liệu phục vụ nghiên cứu.

data/
├── raw/
├── processed/
├── splits/
└── external/
data/raw/

Chứa dữ liệu gốc được tải từ dataset.

Dữ liệu trong thư mục này không nên bị chỉnh sửa trực tiếp.

Ví dụ:

data/raw/
├── images/
├── labels.csv
└── metadata.csv

Mục đích:

Giữ nguyên dữ liệu gốc.
Có thể tái lập preprocessing.
Có thể quay lại dữ liệu ban đầu khi cần kiểm tra.

### data/processed/

Chứa dữ liệu sau preprocessing nếu preprocessing tạo ra file vật lý.

Ví dụ:

data/processed/
├── resized/
├── normalized/
└── metadata.csv

Không bắt buộc phải lưu toàn bộ ảnh đã preprocess nếu pipeline có thể preprocessing trực tiếp trong DataLoader.

### data/splits/

Chứa thông tin chia dataset:

train.csv
val.csv
test.csv

Ví dụ:

image_path,label,patient_id
images/001.jpg,PNEUMONIA,P001
images/002.jpg,NORMAL,P002
Quy tắc quan trọng

Nếu dataset có thông tin bệnh nhân, ưu tiên sử dụng:

PATIENT-LEVEL SPLIT

thay vì chia ngẫu nhiên từng ảnh.

Mục tiêu là tránh trường hợp ảnh của cùng một bệnh nhân xuất hiện đồng thời ở train và test.

### data/external/

Chứa dữ liệu bên ngoài dataset chính nếu cần.

Ví dụ:

External validation dataset.
Additional metadata.
Dataset dùng cho kiểm tra khả năng generalization.
## 6.2. notebooks/

Chứa các notebook dùng cho:

Exploration.
Phân tích dữ liệu.
Prototype.
Visualization.
Kiểm tra kết quả.

Thứ tự notebook phản ánh workflow nghiên cứu:

01 → Data Exploration
02 → Preprocessing
03 → EfficientNet
04 → ViT
05 → Fixed Ensemble
06 → Adaptive Ensemble
07 → Explainability
08 → Final Evaluation
Quy tắc

Notebook dùng để:

Khám phá.
Phân tích.
Visualization.
Thử nghiệm nhanh.

Code dùng lại nhiều lần nên được đưa vào:

### src/

Không nên biến notebook thành nơi chứa toàn bộ source code của project.

## 6.3. src/data/

Chứa logic xử lý dữ liệu.

src/data/
├── dataset.py
├── dataloader.py
├── preprocessing.py
├── augmentation.py
└── split.py
### dataset.py

Định nghĩa Dataset class.

Nhiệm vụ:

Load image.
Load label.
Apply transform.
Return data cho model.
### dataloader.py

Định nghĩa DataLoader.

Quản lý:

Batch size.
Shuffle.
Number of workers.
Train / validation / test loader.
### preprocessing.py

Chứa preprocessing:

Resize.
Normalize.
Convert image format.
Các bước tiền xử lý khác.

Preprocessing cần được thiết kế phù hợp với mô hình và dataset.

### augmentation.py

Chứa data augmentation cho training.

Ví dụ:

Horizontal flip nếu phù hợp.
Rotation nhẹ.
Crop.
Affine transformation.
Các augmentation khác sau khi kiểm tra tính hợp lý với ảnh X-quang.

Không áp dụng augmentation một cách tùy tiện.

### split.py

Chứa logic chia dataset.

Đặc biệt xử lý:

Patient-level split

nếu metadata cung cấp patient ID.

## 6.4. src/models/

Chứa kiến trúc model.

src/models/
├── efficientnet.py
├── vit.py
├── factory.py
└── weights.py
### efficientnet.py

Định nghĩa EfficientNet-B4.

Ví dụ:

Input
  ↓
EfficientNet-B4 Backbone
  ↓
Classification Head
  ↓
Pneumonia Probability

File này chỉ tập trung vào architecture và forward logic.

Không chứa toàn bộ training pipeline.

### vit.py

Định nghĩa Vision Transformer ViT-B/16.

Input
  ↓
Patch Embedding
  ↓
Transformer Encoder
  ↓
Classification Head
  ↓
Pneumonia Probability
### factory.py

Dùng để khởi tạo model theo config.

Ví dụ:

model_name = "efficientnet_b4"

hoặc:

model_name = "vit_b16"
### weights.py

Quản lý việc:

Load pretrained weights.
Load trained checkpoint.
Save checkpoint metadata.
## 6.5. src/training/

Chứa toàn bộ logic training.

src/training/
├── trainer.py
├── train_efficientnet.py
├── train_vit.py
├── losses.py
├── optimizer.py
├── scheduler.py
└── callbacks.py
### trainer.py

Chứa training loop dùng chung.

Quản lý:

Epoch
   ↓
Forward
   ↓
Loss
   ↓
Backward
   ↓
Optimizer
   ↓
Validation
   ↓
Checkpoint
### train_efficientnet.py

Cấu hình training cho EfficientNet-B4.

### train_vit.py

Cấu hình training cho ViT-B/16.

### losses.py

Định nghĩa loss function.

Có thể nghiên cứu:

Cross Entropy.
Weighted Cross Entropy.
Các loss khác nếu cần.

Việc sử dụng loss nào cần được ghi nhận trong experiment configuration.

### optimizer.py

Quản lý optimizer.

Ví dụ:

Adam.
AdamW.
scheduler.py

Quản lý learning rate scheduler.

### callbacks.py

Các callback:

Early stopping.
Model checkpoint.
Logging.
Learning rate tracking.
## 6.6. src/ensemble/

Đây là module quan trọng nhất của đề tài.

src/ensemble/
├── fixed_ensemble.py
├── adaptive_ensemble.py
├── weighting.py
└── uncertainty.py
### fixed_ensemble.py

Chứa các phương pháp ensemble cố định.

Ví dụ:

P_final =
    α × P_CNN
    +
    (1 - α) × P_ViT

Có thể thực nghiệm nhiều giá trị α.

Ví dụ:

α = 0.1
α = 0.2
...
α = 0.9

Việc lựa chọn α cuối cùng phải dựa trên validation set, không tối ưu trực tiếp trên test set.

### adaptive_ensemble.py

Chứa logic ensemble thích ứng.

Khác với fixed ensemble:

Fixed:

α = constant

Adaptive:

α = f(input, prediction, confidence, uncertainty, ...)

Tức là trọng số có thể thay đổi theo từng ảnh.

### weighting.py

Chứa các chiến lược tính trọng số.

Ví dụ:

confidence-based weighting

hoặc các phương pháp khác được nhóm nghiên cứu và lựa chọn.

### uncertainty.py

Chứa logic ước lượng uncertainty.

Ví dụ các phương pháp có thể được nghiên cứu:

Entropy-based uncertainty.
Predictive uncertainty.
Các phương pháp uncertainty estimation phù hợp với kiến trúc được sử dụng.

Phương pháp cuối cùng cần được thống nhất trước khi implementation.

## 6.7. src/evaluation/

Chứa toàn bộ logic đánh giá.

src/evaluation/
├── metrics.py
├── evaluator.py
├── roc.py
├── confusion_matrix.py
└── calibration.py
### metrics.py

Tính:

Accuracy.
Precision.
Recall.
F1-score.
Sensitivity.
Specificity.
ROC-AUC.
### evaluator.py

Chạy evaluation hoàn chỉnh cho một model hoặc ensemble.

Ví dụ:

EfficientNet-B4
ViT-B/16
Fixed Ensemble
Adaptive Ensemble

và trả về một bộ metrics thống nhất.

### roc.py

Tạo:

ROC curve.
AUC.
So sánh ROC giữa các model.
confusion_matrix.py

Tạo confusion matrix.

Dùng để phân tích:

True Positive
True Negative
False Positive
False Negative
calibration.py

Phục vụ nghiên cứu mức độ tin cậy của probability.

Có thể chứa:

Reliability diagram.
Expected Calibration Error.
Temperature scaling nếu được sử dụng.

Module này đặc biệt hữu ích nếu Adaptive Ensemble dựa trên confidence.

## 6.8. src/explainability/

Chứa Explainable AI.

src/explainability/
├── gradcam.py
├── vit_explain.py
├── ensemble_explain.py
└── visualization.py
### gradcam.py

Triển khai Grad-CAM hoặc phương pháp tương đương cho EfficientNet-B4.

Output:

Original X-ray
      +
Grad-CAM Heatmap
      ↓
Visualization
vit_explain.py

Chứa phương pháp visualization phù hợp với Vision Transformer.

Ví dụ:

Attention visualization.
Attention rollout.
Hoặc phương pháp XAI khác được lựa chọn.
### ensemble_explain.py

Nghiên cứu cách giải thích quyết định cuối cùng của ensemble.

Ví dụ:

EfficientNet explanation
        +
ViT explanation
        +
Adaptive weights
        ↓
Final explanation

Phạm vi cụ thể cần được thống nhất trong quá trình nghiên cứu.

### visualization.py

Chứa các hàm:

Overlay heatmap.
Resize heatmap.
Save visualization.
Tạo hình phục vụ luận văn.
## 6.9. src/inference/

Chứa pipeline inference được sử dụng khi chạy model ngoài training.

src/inference/
├── predictor.py
├── preprocessing.py
└── postprocessing.py
### predictor.py

Load model và thực hiện prediction.

Ví dụ:

Input X-ray
    ↓
Preprocessing
    ↓
EfficientNet
    ↓
ViT
    ↓
Ensemble
    ↓
Prediction
### preprocessing.py

Preprocessing riêng cho inference.

Điều quan trọng là preprocessing inference phải nhất quán với preprocessing đã sử dụng khi validation/test.

### postprocessing.py

Xử lý output:

Probability.
Predicted class.
Confidence.
Ensemble weights.
Các thông tin cần trả về frontend.
## 6.10. src/utils/

Các tiện ích dùng chung.

src/utils/
├── seed.py
├── logger.py
├── device.py
├── checkpoint.py
└── visualization.py

Ví dụ:

### seed.py

Đảm bảo reproducibility.

### logger.py

Logging training và experiment.

### device.py

Quản lý:

CPU
CUDA
GPU
### checkpoint.py

Save / load checkpoint.

## 6.11. configs/

Chứa configuration của project.

configs/
├── dataset.yaml
├── efficientnet_b4.yaml
├── vit_b16.yaml
├── fixed_ensemble.yaml
├── adaptive_ensemble.yaml
└── experiment.yaml

Mục đích:

Không hard-code hyperparameter trực tiếp vào source code.

Ví dụ:

model:
  name: efficientnet_b4
  pretrained: true
  num_classes: 2

training:
  batch_size: 16
  epochs: 30
  learning_rate: 0.0001

data:
  image_size: 380

Các hyperparameter cụ thể chỉ là ví dụ và sẽ được xác định trong quá trình thực nghiệm.

## 6.12. experiments/

Đây là thư mục quản lý các thí nghiệm nghiên cứu.

experiments/
├── baseline/
│   ├── efficientnet_b4/
│   └── vit_b16/
├── fixed_ensemble/
├── adaptive_ensemble/
└── ablation/
experiments/baseline/

Chứa các experiment baseline.

### efficientnet_b4/
### vit_b16/

Mục đích:

Xác định hiệu năng của từng model khi hoạt động độc lập.

experiments/fixed_ensemble/

Chứa các thí nghiệm fixed ensemble.

Ví dụ:

α = 0.5
α = 0.6
α = 0.7
...
### experiments/adaptive_ensemble/

Chứa các experiment adaptive ensemble.

Ví dụ:

### confidence_based/
### uncertainty_based/
### experiments/ablation/

Dùng cho ablation study.

Ví dụ:

Without CNN
Without ViT
Without adaptive weighting
Confidence only
Uncertainty only

Mục tiêu là kiểm tra thành phần nào thực sự đóng góp vào hiệu quả của phương pháp.

## 6.13. models/

Chứa trained model weights.

Khác với:

src/models/

src/models/ chứa code kiến trúc.

models/ chứa model đã train.

Ví dụ:

models/
├── efficientnet_b4/
│   ├── best.pt
│   └── config.yaml
│
├── vit_b16/
│   ├── best.pt
│   └── config.yaml
│
└── ensemble/
    └── config.yaml

Không commit các checkpoint quá lớn vào Git nếu project sử dụng GitHub.

Có thể sử dụng Git LFS hoặc lưu model ở storage riêng.

## 6.14. results/

Chứa toàn bộ kết quả thực nghiệm.

results/
├── metrics/
├── plots/
├── roc_curves/
├── confusion_matrices/
├── calibration/
├── heatmaps/
└── tables/
### results/metrics/

Ví dụ:

efficientnet_b4.json
vit_b16.json
fixed_ensemble.json
adaptive_ensemble.json
### results/plots/

Các biểu đồ:

Training loss.
Validation loss.
Accuracy curve.
F1 curve.
Weight distribution.
### results/roc_curves/

Chứa ROC curve.

Ví dụ:

model_comparison.png
### results/confusion_matrices/

Confusion matrix của từng phương pháp.

### results/calibration/

Các biểu đồ và kết quả calibration.

### results/heatmaps/

Các hình visualization:

Original
Grad-CAM
Attention
Overlay
### results/tables/

Các bảng dùng trong báo cáo/luận văn.

Ví dụ:

model_comparison.csv
ensemble_comparison.csv
ablation_results.csv
## 6.15. api/

Backend phục vụ inference.

api/
├── routes/
├── schemas/
├── services/
└── main.py
### api/routes/

Định nghĩa API endpoint.

Ví dụ:

POST /predict
GET  /health
api/schemas/

Định nghĩa input/output schema.

Ví dụ response:

{
  "prediction": "PNEUMONIA",
  "probability": 0.934,
  "confidence": 0.934,
  "cnn_probability": 0.88,
  "vit_probability": 0.95,
  "cnn_weight": 0.32,
  "vit_weight": 0.68
}

Các field cuối cùng sẽ được quyết định khi thiết kế Adaptive Ensemble.

### api/services/

Chứa business logic.

#### inference_service.py

Thực hiện prediction.

#### explainability_service.py

Sinh heatmap / explanation.

### api/main.py

Entry point của backend.

## 6.16. frontend/

Chứa giao diện web demo.

frontend/
└── app.py

Có thể sử dụng Streamlit cho phiên bản demo của khóa luận.

Giao diện dự kiến:

+------------------------------------------------+
|        PNEUMONIA DETECTION SYSTEM              |
+------------------------------------------------+
|                                                |
|          [ Upload Chest X-ray ]                |
|                                                |
|                X-RAY IMAGE                     |
|                                                |
+------------------------------------------------+
| Prediction:      PNEUMONIA                     |
| Probability:     93.4%                         |
| Confidence:      High                          |
+------------------------------------------------+
| EfficientNet:    88%                           |
| ViT:             95%                           |
|                                                |
| Ensemble weights:                              |
| CNN:             32%                           |
| ViT:             68%                           |
+------------------------------------------------+
|                  Grad-CAM                      |
|                                                |
|              [ HEATMAP ]                       |
|                                                |
+------------------------------------------------+

## 6.17. scripts/

Các script chạy từ command line.

scripts/
├── prepare_dataset.py
├── train_efficientnet.py
├── train_vit.py
├── evaluate_models.py
└── generate_heatmaps.py

Ví dụ:

python scripts/prepare_dataset.py
python scripts/train_efficientnet.py
python scripts/train_vit.py
python scripts/evaluate_models.py
python scripts/generate_heatmaps.py

## 6.18. tests/

Unit tests và integration tests.

tests/
├── test_dataset.py
├── test_models.py
├── test_ensemble.py
└── test_inference.py

Mục tiêu:

Kiểm tra Dataset.
Kiểm tra model output.
Kiểm tra ensemble.
Kiểm tra inference pipeline.

Ví dụ:

Input image
    ↓
Prediction probability
    ↓
0 <= probability <= 1

## 6.19. requirements.txt

Danh sách dependency của project.

Ví dụ có thể bao gồm:

torch
torchvision
transformers
scikit-learn
numpy
pandas
matplotlib
opencv-python
Pillow
PyYAML
streamlit
fastapi
uvicorn

Danh sách chính thức sẽ được chốt sau khi thống nhất framework.

## 6.20. .env

Chứa environment variables.

Ví dụ:

MODEL_PATH=...
DEVICE=cuda
API_HOST=...
API_PORT=...

Không commit .env chứa thông tin nhạy cảm.

## 6.21. .gitignore

Không commit:

data/raw/
data/processed/
models/*.pt
.env
__pycache__/
.ipynb_checkpoints/

tùy chiến lược quản lý dữ liệu và model của nhóm.

---

# 7. Model Strategy
## 7.1. EfficientNet-B4

Vai trò:
CNN Baseline
Mục đích:
- Đánh giá hiệu quả của CNN.
- Là một nhánh trong ensemble.
- Sinh confidence/probability.
- Hỗ trợ explainability bằng Grad-CAM.

## 7.2. ViT-B/16

Vai trò:
Transformer Baseline
Mục đích:
- Đánh giá Vision Transformer.
- Là nhánh thứ hai của ensemble.
- Sinh probability/confidence.
- Nghiên cứu explainability phù hợp với Transformer.

--- 

# 8. Ensemble Strategy
## 8.1. Individual Models

Đầu tiên đánh giá:
- EfficientNet-B4
- ViT-B/16
Chưa ensemble.

Mục tiêu:
Biết được khả năng của từng mô hình độc lập.

## 8.2. Fixed Ensemble
Ví dụ:
P_CNN  = EfficientNet prediction
P_ViT  = ViT prediction
P_final =
    α × P_CNN
    +
    (1 - α) × P_ViT
Trọng số:
α = constant
trong toàn bộ quá trình inference.

---

## 8.3. Adaptive Ensemble

Adaptive Ensemble thay đổi trọng số theo từng input.
Concept:

                  Input X-ray
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    EfficientNet                  ViT
          │                         │
          ▼                         ▼
     Prediction                 Prediction
          │                         │
          ▼                         ▼
     Confidence /             Confidence /
     Uncertainty              Uncertainty
          │                         │
          └────────────┬────────────┘
                       ▼
                 Weighting Module
                       │
                       ▼
                Adaptive Fusion
                       │
                       ▼
                 Final Prediction

Công thức cuối cùng sẽ được xác định sau khi nhóm hoàn tất nghiên cứu literature và thiết kế phương pháp.

---

# 9. Evaluation Strategy

Tất cả các phương pháp phải được đánh giá trên cùng một test set.
So sánh:

Model	Accuracy	Precision	Recall	F1	ROC-AUC	Sensitivity	Specificity
EfficientNet-B4							
ViT-B/16							
Fixed Ensemble							
Adaptive Ensemble							

Không lựa chọn model tốt nhất chỉ dựa trên Accuracy.

Đối với bài toán y tế cần đặc biệt quan tâm:
- Sensitivity.
- Specificity.
- Recall.
- ROC-AUC.
- F1-score.

---

# 10. Explainability Strategy

EfficientNet-B4
Sử dụng:
## Grad-CAM

Flow:

X-ray
  ↓
EfficientNet
  ↓
Prediction
  ↓
Grad-CAM
  ↓
Heatmap
  ↓
Overlay

## ViT-B/16

Nghiên cứu phương pháp phù hợp với Transformer, chẳng hạn attention-based visualization.

## Ensemble
Nếu khả thi, nghiên cứu cách biểu diễn explanation của quyết định ensemble.

Mục tiêu cuối cùng:
X-ray
   +
Prediction
   +
Confidence
   +
Important Region

để người dùng có thể quan sát vùng ảnh mà mô hình tập trung.

---

# 11. Experimental Design

Đề tài dự kiến gồm các nhóm thực nghiệm:
## Experiment 1 — EfficientNet Baseline
Dataset
   ↓
EfficientNet-B4
   ↓
Evaluation
## Experiment 2 — ViT Baseline
Dataset
   ↓
ViT-B/16
   ↓
Evaluation
## Experiment 3 — Fixed Ensemble
EfficientNet + ViT
        ↓
Fixed Weight
        ↓
Evaluation
## Experiment 4 — Confidence-based Adaptive Ensemble
EfficientNet + ViT
        ↓
Confidence
        ↓
Adaptive Weight
        ↓
Evaluation
## Experiment 5 — Uncertainty-based Adaptive Ensemble
EfficientNet + ViT
        ↓
Uncertainty
        ↓
Adaptive Weight
        ↓
Evaluation
## Experiment 6 — Ablation Study
Kiểm tra ảnh hưởng của từng thành phần.
Ví dụ:
Full Adaptive Ensemble
        ↓
Remove adaptive weighting
        ↓
Remove uncertainty
        ↓
Remove CNN
        ↓
Remove ViT

Mục tiêu là chứng minh thành phần đề xuất thực sự có đóng góp.

---

# 12. Reproducibility

Mỗi experiment cần lưu:
- Configuration
- Model
- Dataset split
- Random seed
- Training parameters
- Best checkpoint
- Metrics
- Plots
Ví dụ:
experiments/adaptive_ensemble/confidence_based/
├── config.yaml
├── results.json
├── metrics.csv
└── notes.md

Mục tiêu:
Một experiment đã chạy phải có khả năng được tái tạo lại.

---

# 13. Data Leakage Prevention

Đây là yêu cầu quan trọng đối với bài toán Medical Imaging.
Nếu dataset có patient identifier, không được để ảnh của cùng một bệnh nhân xuất hiện đồng thời ở:
- Train
- Validation
- Test
Ví dụ không hợp lệ:
Patient A
├── image_01 → Train
├── image_02 → Test
└── image_03 → Validation

Ưu tiên:
Patient A → Train
Patient B → Validation
Patient C → Test

Ngoài ra cần đảm bảo:
- Không sử dụng test set để tuning hyperparameter.
- Không chọn ensemble weight dựa trực tiếp trên test set.
- Không chọn threshold dựa trực tiếp trên test set.
- Test set chỉ được sử dụng để đánh giá cuối cùng.

---

# 14. Class Imbalance

Nếu dataset bị mất cân bằng giữa:
- NORMAL
- PNEUMONIA
cần phân tích trước khi training.
Có thể cân nhắc:
- Class weight.
- Weighted loss.
- Balanced sampling.
- Appropriate augmentation.

Không nên áp dụng tất cả phương pháp cùng lúc mà không có experiment.
Mỗi kỹ thuật cần được ghi nhận và đánh giá rõ ràng.

---

# 15. Configuration Management

Hyperparameter không nên hard-code trong nhiều file.
Ví dụ:
configs/
├── efficientnet_b4.yaml
├── vit_b16.yaml
├── fixed_ensemble.yaml
└── adaptive_ensemble.yaml

Mục tiêu:
Configuration
      ↓
Training
      ↓
Experiment
      ↓
Result

Có thể dễ dàng thay đổi:
1. Batch size.
2. Learning rate.
3. Epoch.
4. Image size.
5. Optimizer.
6. Scheduler.
7. Ensemble weight.
8. Threshold.

---

# 16. Naming Convention

## Python files

Sử dụng:
- snake_case.py
Ví dụ:
adaptive_ensemble.py
efficientnet.py
data_loader.py

## Classes

Sử dụng:
- PascalCase
Ví dụ:
EfficientNetB4Classifier
ViTB16Classifier
AdaptiveEnsemble
Functions

Sử dụng:
- snake_case
Ví dụ:
load_model()
evaluate_model()
compute_metrics()
calculate_weights()

# 17. Git Workflow

Khuyến nghị sử dụng:

- main
- develop
- feature/*

Ví dụ:
feature/data-preprocessing
feature/efficientnet-baseline
feature/vit-baseline
feature/fixed-ensemble
feature/adaptive-ensemble
feature/explainability
feature/web-demo

Không commit trực tiếp mọi thay đổi lớn vào main.

# 18. Development Workflow

Thứ tự triển khai dự kiến:
Phase 1
│
├── Dataset
├── Data exploration
└── Data preprocessing
        ↓
Phase 2
│
├── EfficientNet-B4
└── Baseline evaluation
        ↓
Phase 3
│
├── ViT-B/16
└── Baseline evaluation
        ↓
Phase 4
│
└── Fixed Ensemble
        ↓
Phase 5
│
├── Confidence-based Ensemble
└── Uncertainty-based Ensemble
        ↓
Phase 6
│
├── Evaluation
├── Calibration
└── Ablation Study
        ↓
Phase 7
│
└── Explainability
        ↓
Phase 8
│
├── Best Model
├── Inference pipeline
└── Web Demo

---

# 19. Nguyên tắc quan trọng của project
## Rule 1 — Không code trước khi thống nhất data split

Dataset split phải được xác định trước.

## Rule 2 — Hai model phải được đánh giá công bằng

EfficientNet-B4 và ViT-B/16 cần sử dụng cùng:
- Dataset split.
- Evaluation protocol.
- Test set.
Các preprocessing đặc thù kiến trúc vẫn có thể khác nhau nếu cần thiết.

## Rule 3 — Không tune trên test set

Test set chỉ sử dụng cho đánh giá cuối cùng.

## Rule 4 — Không chỉ báo cáo Accuracy

Phải báo cáo nhiều metric:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Sensitivity
- Specificity
## Rule 5 — Mọi experiment phải có configuration

Không chạy experiment quan trọng mà không lưu:
- config
- seed
- model
- checkpoint
- metrics
## Rule 6 — Không đưa business logic vào notebook

Notebook chủ yếu dành cho:
- Exploration
- Visualization
- Analysis
Reusable code đưa vào: src/
## Rule 7 — Không coi confidence = uncertainty một cách mặc định

Probability cao không nhất thiết đồng nghĩa với mô hình được calibration tốt hoặc uncertainty thấp.
Nếu Adaptive Ensemble sử dụng confidence hoặc uncertainty, định nghĩa và phương pháp tính phải được nêu rõ trong luận văn.

## Rule 8 — Web chỉ là lớp demo

Không để phần frontend/backend chiếm phần lớn thời gian nghiên cứu.
Ưu tiên:
Research > Experiment > Evaluation > Explainability > Deployment

---

# 20. Expected Final System

Hệ thống cuối cùng dự kiến:

                       USER
                        │
                        ▼
                  Upload X-ray
                        │
                        ▼
                 Preprocessing
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        EfficientNet-B4         ViT-B/16
              │                   │
              ▼                   ▼
         Probability          Probability
              │                   │
              ▼                   ▼
         Confidence /         Confidence /
         Uncertainty          Uncertainty
              │                   │
              └─────────┬─────────┘
                        ▼
                 Adaptive Ensemble
                        │
                        ▼
                Final Prediction
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
         Class       Confidence   Weight
             │          │          │
             └──────────┼──────────┘
                        ▼
                 Explainability
                        │
                        ▼
                  Heatmap / XAI
                        │
                        ▼
                  Web Interface

---

# 21. Expected Web Demo

Ứng dụng web dự kiến hiển thị:
## Input

Chest X-ray image
## Output

Prediction: PNEUMONIA / NORMAL
Probability: 93.4%
Confidence: High
EfficientNet probability: 88%
ViT probability: 95%
Adaptive weights:
CNN: 32%
ViT: 68%
## Explainability

Original X-ray
       +
Heatmap
       ↓
Suspected region visualization

Đây là hệ thống minh họa nghiên cứu, không được sử dụng thay thế cho chẩn đoán của bác sĩ.

---

# 22. Expected Research Results

Bảng kết quả cuối cùng dự kiến:

Method	           Accuracy	  Precision	 Recall	 F1	ROC-AUC	   Sensitivity	Specificity
EfficientNet-B4	   TBD	      TBD	     TBD	 TBD	TBD	   TBD	            TBD
ViT-B/16	       TBD	      TBD	     TBD	 TBD	TBD	   TBD	            TBD
Fixed Ensemble	   TBD	      TBD	     TBD	 TBD	TBD	   TBD	            TBD
Adaptive Ensemble  TBD	      TBD	     TBD	 TBD	TBD	   TBD	            TBD
TBD sẽ được thay thế sau khi thực nghiệm.

Không đặt trước kết quả để tránh bias trong quá trình nghiên cứu.

---

# 23. Expected Contribution

Đề tài hướng tới các đóng góp:
Contribution 1
- Xây dựng baseline CNN bằng EfficientNet-B4.

Contribution 2
- Xây dựng baseline Vision Transformer bằng ViT-B/16.

Contribution 3
- Xây dựng và đánh giá Fixed Ensemble.

Contribution 4
- Đề xuất và thực nghiệm Adaptive Ensemble dựa trên confidence hoặc uncertainty.

Contribution 5
- So sánh các phương pháp trên cùng một protocol.

Contribution 6
- Sử dụng Explainable AI để trực quan hóa quyết định của mô hình.

Contribution 7
- Xây dựng web application minh họa mô hình tốt nhất.