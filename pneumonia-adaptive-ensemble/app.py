import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.models.factory import build_model
from src.ensemble.fixed_ensemble import FixedEnsemble
from src.ensemble.adaptive_ensemble import AdaptiveEnsemble
from src.explainability.gradcam import generate_gradcam_heatmap
from torchvision import transforms

st.set_page_config(
    page_title="Chẩn đoán Viêm Phổi X-quang | Đồ án tốt nghiệp",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #3B82F6;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9CA3AF;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-card-pneumonia {
        background-color: #451A1A;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #FCA5A5;
    }
    .result-card-normal {
        background-color: #064E3B;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #A7F3D0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_single_model(name, rel_path, device):
    full_path = os.path.join(BASE_DIR, rel_path)
    model = None
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        try:
            model = build_model(name, num_classes=2, pretrained=False).to(device)
            model.load_state_dict(torch.load(full_path, map_location=device, weights_only=True))
            model.eval()
            return model
        except Exception:
            pass
            
    try:
        model = build_model(name, num_classes=2, pretrained=True).to(device)
        model.eval()
        return model
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_all_models():
    device = get_device()
    checkpoint_paths = {
        'EfficientNet-B4': 'models/efficientnet_b4/best.pt',
        'ViT-B/16': 'models/vit_b16/best.pt',
        'ResNet50': 'models/resnet50/best.pt'
    }
    models = {}
    for name, path in checkpoint_paths.items():
        m = load_single_model(name, path, device)
        if m is not None:
            models[name] = m
    return models, device

def transform_image(pil_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0)

def predict_probabilities(_models_dict, _img_tensor):
    probs = {}
    with torch.no_grad():
        for name, model in _models_dict.items():
            outputs = model(_img_tensor)
            prob_pneu = torch.softmax(outputs, dim=1)[0, 1].item()
            probs[name] = prob_pneu
    return probs

def main():
    st.markdown('<div class="main-header">🫁 HỆ THỐNG CHẨN ĐOÁN VIÊM PHỔI TỪ ẢNH X-QUANG NGỰC</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mô hình Ensemble Cố định & Thích ứng CNN–ViT | Mã đề tài: <b>CNTT-KLCN171</b></div>', unsafe_allow_html=True)

    st.sidebar.image("https://img.icons8.com/color/96/lungs.png", width=70)
    st.sidebar.title("🛠️ Cấu hình Chẩn đoán")
    st.sidebar.markdown("**GVHD:** TS. Phùng Thế Bảo")
    st.sidebar.markdown("**Nhóm SV:** Nguyễn Gia Khang (NT), Phan Minh Tâm, Trịnh Minh Hiếu")
    st.sidebar.divider()

    model_option = st.sidebar.selectbox(
        "Lựa chọn Mô hình Chẩn đoán:",
        [
            "⚡ Fixed Soft Voting Ensemble (EffNet + ViT)",
            "🔒 Fixed Hard Voting Ensemble (EffNet + ViT)",
            "🎯 Adaptive Ensemble (Confidence Weighted)",
            "🔹 EfficientNet-B4 (Single)",
            "🔹 ViT-B/16 (Single)",
            "🔹 ResNet50 (Single)"
        ]
    )

    threshold = st.sidebar.slider("Ngưỡng phân loại Pneumonia:", 0.1, 0.9, 0.5, 0.05)
    st.sidebar.info("💡 Ngưỡng mặc định 0.50. Tăng/giảm ngưỡng để cân bằng giữa Sensitivity (Độ nhạy) và Specificity.")

    tab1, tab2, tab3 = st.tabs(["🩺 Chẩn đoán & Giải thích Grad-CAM", "📊 Bảng Kết quả Thực nghiệm", "🛠️ Hướng dẫn Mở rộng / Đổi Mô hình"])

    models, device = load_all_models()

    with tab1:
        st.subheader("📤 Tải lên ảnh X-quang Ngực (Chest X-Ray)")
        
        uploaded_file = st.file_uploader("Chọn file ảnh X-quang (.png, .jpg, .jpeg):", type=["png", "jpg", "jpeg"])
        
        active_img = None
        if uploaded_file is not None:
            active_img = Image.open(uploaded_file).convert("RGB")

        if active_img is not None:
            col_img, col_res = st.columns([1, 1.2])
            
            with col_img:
                st.image(active_img, caption="Ảnh X-quang đang chẩn đoán", use_container_width=True)

            with col_res:
                st.subheader("🔍 Kết quả Chẩn đoán")
                
                with st.spinner("Đang chẩn đoán ảnh X-quang..."):
                    img_tensor = transform_image(active_img).to(device)
                    probs_dict = predict_probabilities(models, img_tensor)
                    
                    p_eff = probs_dict.get('EfficientNet-B4', 0.5)
                    p_vit = probs_dict.get('ViT-B/16', 0.5)
                    p_res = probs_dict.get('ResNet50', 0.5)

                    if "Soft Voting" in model_option:
                        final_prob = (p_eff + p_vit) / 2.0
                    elif "Hard Voting" in model_option:
                        pred1 = 1.0 if p_eff >= threshold else 0.0
                        pred2 = 1.0 if p_vit >= threshold else 0.0
                        final_prob = (pred1 + pred2) / 2.0
                    elif "Adaptive" in model_option:
                        p_list = [p_eff, p_vit, p_res]
                        final_prob = float(AdaptiveEnsemble(strategy='confidence').predict_probs(np.array(p_list)[:, None])[0])
                    elif "EfficientNet" in model_option:
                        final_prob = p_eff
                    elif "ViT" in model_option:
                        final_prob = p_vit
                    else:
                        final_prob = p_res

                    is_pneumonia = final_prob >= threshold

                    if is_pneumonia:
                        st.markdown(f"""
                        <div class="result-card-pneumonia">
                            <h3 style="color: #EF4444; margin:0;">🔴 CHẨN ĐOÁN: VIÊM PHỔI (PNEUMONIA)</h3>
                            <p style="margin-top:0.5rem; color:#FCA5A5;">Độ tin cậy mắc bệnh: <b>{final_prob*100:.2f}%</b> (Ngưỡng: {threshold:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card-normal">
                            <h3 style="color: #10B981; margin:0;">🟢 CHẨN ĐOÁN: BÌNH THƯỜNG (NORMAL)</h3>
                            <p style="margin-top:0.5rem; color:#A7F3D0;">Độ tin cậy khỏe mạnh: <b>{(1-final_prob)*100:.2f}%</b> (Ngưỡng: {threshold:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.progress(float(np.clip(final_prob, 0.0, 1.0)))
                    st.caption(f"Xác suất Viêm phổi: {final_prob:.4f} | Bình thường: {1-final_prob:.4f}")

                    st.divider()
                    st.markdown("##### 📊 Dự đoán chi tiết từng mô hình:")
                    m_df = pd.DataFrame([
                        {"Mô hình": "EfficientNet-B4", "Xác suất Pneumonia": f"{p_eff*100:.2f}%"},
                        {"Mô hình": "ViT-B/16", "Xác suất Pneumonia": f"{p_vit*100:.2f}%"},
                        {"Mô hình": "ResNet50", "Xác suất Pneumonia": f"{p_res*100:.2f}%"}
                    ])
                    st.dataframe(m_df, hide_index=True, use_container_width=True)

            # Grad-CAM Section
            st.divider()
            st.subheader("🔍 Bản đồ Nhiệt Giải thích Vùng Tổn thương (Grad-CAM)")
            
            with st.spinner("Đang trích xuất bản đồ nhiệt Grad-CAM..."):
                cam_model = models.get('ResNet50', models.get('EfficientNet-B4'))
                if cam_model is not None:
                    img_tensor_cam = transform_image(active_img).to(device)
                    heatmap_np = generate_gradcam_heatmap(cam_model, img_tensor_cam)

                    orig_np = np.array(active_img.resize((224, 224))) / 255.0
                    cam_colored = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
                    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB) / 255.0
                    overlay = 0.6 * orig_np + 0.4 * cam_colored
                    overlay = np.clip(overlay, 0, 1)

                    col_c1, col_c2, col_c3 = st.columns(3)
                    with col_c1:
                        st.image(orig_np, caption="Ảnh Gốc (224x224)", use_container_width=True)
                    with col_c2:
                        st.image(heatmap_np, caption="Bản đồ nhiệt Grad-CAM", use_container_width=True)
                    with col_c3:
                        st.image(overlay, caption="Chồng phủ chú ý (Overlay)", use_container_width=True)

    with tab2:
        st.subheader("📈 Bảng Tổng hợp Kết quả Thực nghiệm (Test Set 624 ảnh)")
        csv_file = os.path.join(BASE_DIR, 'results', 'experiment_results.csv')
        if os.path.exists(csv_file):
            df_res = pd.read_csv(csv_file)
            st.dataframe(df_res.style.highlight_max(axis=0, color='#064E3B'), use_container_width=True)
        else:
            st.info("Chạy `python scripts/evaluate_models.py` để xuất bảng kết quả CSV chi tiết.")

        col_f1, col_f2 = st.columns(2)
        roc_p = os.path.join(BASE_DIR, 'results', 'figures', 'roc_curves.png')
        cm_p = os.path.join(BASE_DIR, 'results', 'figures', 'confusion_matrices.png')
        with col_f1:
            if os.path.exists(roc_p):
                st.image(roc_p, caption="Hình 1: Đường cong ROC so sánh hiệu năng", use_container_width=True)
        with col_f2:
            if os.path.exists(cm_p):
                st.image(cm_p, caption="Hình 2: Ma trận nhầm lẫn (Confusion Matrix)", use_container_width=True)

    with tab3:
        st.subheader("🛠️ Hướng dẫn Thay đổi / Mở rộng Mô hình Thực nghiệm")
        st.markdown("""
        Hệ thống được thiết kế theo kiến trúc **Modular Factory**, cho phép bạn thay đổi hoặc thêm bất kỳ kiến trúc mô hình mới nào (từ thư viện `timm`) chỉ bằng cách thay đổi tên mô hình!

        #### 1. Đổi hoặc thêm mô hình trong mã nguồn (`scripts/train_models.py`):
        Để thử nghiệm mô hình mới (ví dụ: `DenseNet-121`, `Swin Transformer`, `ConvNeXt`):
        
        ```python
        # Trong scripts/train_models.py:
        models_to_train = ['efficientnet_b4', 'vit_b16', 'densenet121', 'convnext_base']
        
        for name in models_to_train:
            train_model(name, epochs=3)
        ```

        #### 2. Thêm alias tên mô hình trong `src/models/factory.py`:
        ```python
        # Trong src/models/factory.py:
        MODEL_ALIAS = {
            'densenet121': 'densenet121',
            'swin_base': 'swin_base_patch4_window7_224',
            'convnext_base': 'convnext_base'
        }
        ```

        #### 3. Đánh giá lại toàn bộ đường ống thực nghiệm:
        ```bash
        python scripts/evaluate_models.py
        python scripts/generate_pdf_report.py
        ```
        """)

if __name__ == '__main__':
    main()
