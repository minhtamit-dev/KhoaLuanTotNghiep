"""Streamlit Web Application for Pneumonia Detection with Adaptive CNN-ViT Ensemble & XAI."""
import io
import sys
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability.visualization import overlay_heatmap_on_image
from src.inference.predictor import PneumoniaPredictor


# Configure Streamlit page
st.set_page_config(
    page_title="Pneumonia AI Diagnostics | Adaptive CNN-ViT Ensemble",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 15px;
    }
    .badge-pneumonia {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #16A34A;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(mode: str = "adaptive_ensemble"):
    """Cache model predictor instance."""
    return PneumoniaPredictor(mode=mode)


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/lungs.png", width=64)
        st.title("⚙️ Cấu hình Hệ thống")
        st.markdown("---")
        
        mode = st.selectbox(
            "Phương pháp suy luận (Model Mode)",
            options=[
                "adaptive_ensemble",
                "fixed_ensemble",
                "efficientnet_b4",
                "vit_b16",
            ],
            format_func=lambda x: {
                "adaptive_ensemble": "🌟 Adaptive Ensemble (Đề xuất)",
                "fixed_ensemble": "⚖️ Fixed Ensemble (α=0.5)",
                "efficientnet_b4": "🖼️ EfficientNet-B4 (CNN Baseline)",
                "vit_b16": "⚡ ViT-B/16 (Transformer Baseline)",
            }[x],
        )

        generate_xai = st.checkbox("Sinh bản đồ nhiệt XAI (Heatmap)", value=True)
        heatmap_alpha = st.slider("Độ mờ Heatmap (Opacity)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

        st.markdown("---")
        st.info("💡 **Ghi chú Khóa luận:**\n\nỨng dụng kết hợp kiến trúc CNN và Vision Transformer nhằm tối ưu hóa độ nhạy (Sensitivity) và giải thích quyết định bằng XAI.")

    # Header
    st.markdown('<div class="main-header">🫁 Hệ thống Nhận diện Viêm phổi từ Ảnh X-quang Ngực</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Nghiên cứu phương pháp Ensemble Thích ứng giữa EfficientNet-B4 và Vision Transformer ViT-B/16</div>', unsafe_allow_html=True)

    predictor = load_predictor(mode=mode)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📥 1. Tải ảnh X-quang")
        uploaded_file = st.file_uploader(
            "Chọn file ảnh X-quang ngực (JPEG / PNG)",
            type=["jpg", "jpeg", "png"],
            help="Tải ảnh X-quang chụp tư thế PA hoặc AP",
        )

        sample_btn = st.button("🧪 Thử với ảnh mẫu", use_container_width=True)

        image_to_process = None
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
        elif sample_btn:
            # Generate sample placeholder chest X-ray
            image_to_process = Image.new("RGB", (380, 380), color=(80, 80, 80))

        if image_to_process is not None:
            st.image(image_to_process, caption="Ảnh X-quang đầu vào", use_container_width=True)

    with col2:
        st.subheader("📊 2. Kết quả Chẩn đoán & Phân tích")
        if image_to_process is not None:
            with st.spinner("Đang xử lý ảnh và chạy mô hình suy luận..."):
                res, xai, pil_img = predictor.predict(image_to_process, generate_xai=generate_xai)

            is_pneumonia = res["prediction"] == "PNEUMONIA"
            badge_class = "badge-pneumonia" if is_pneumonia else "badge-normal"
            badge_text = "NGHI NGỜ VIÊM PHỔI (PNEUMONIA)" if is_pneumonia else "BÌNH THƯỜNG (NORMAL)"

            st.markdown(f"""
            <div class="metric-card">
                <h3>Kết luận: <span class="{badge_class}">{badge_text}</span></h3>
                <p><b>Độ tin cậy:</b> {res['confidence']*100:.1f}% ({res['confidence_level']})</p>
                <p><b>Chiến lược:</b> <code>{res['strategy']}</code></p>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities breakdown
            st.write("##### Xác suất từng lớp:")
            p_pneu = res["probability_pneumonia"]
            p_norm = res["probability_normal"]

            st.progress(p_pneu)
            c1, c2 = st.columns(2)
            c1.metric("Xác suất Viêm phổi", f"{p_pneu*100:.2f}%")
            c2.metric("Xác suất Bình thường", f"{p_norm*100:.2f}%")

            # Branch details if ensemble
            b_details = res["branch_details"]
            if b_details["cnn_weight"] is not None and b_details["vit_weight"] is not None:
                st.write("##### Phân bổ trọng số Ensemble thích ứng (Dynamic Weights):")
                w_cnn = b_details["cnn_weight"]
                w_vit = b_details["vit_weight"]
                
                col_w1, col_w2 = st.columns(2)
                col_w1.info(f"🖼️ **CNN (EfficientNet):** {w_cnn*100:.1f}% (P={b_details['cnn_probability']*100:.1f}%)")
                col_w2.success(f"⚡ **ViT (Transformer):** {w_vit*100:.1f}% (P={b_details['vit_probability']*100:.1f}%)")
        else:
            st.info("👈 Vui lòng tải ảnh X-quang hoặc bấm **'Thử với ảnh mẫu'** ở cột bên trái để bắt đầu.")

    # Explainability Section
    if image_to_process is not None and generate_xai and xai is not None:
        st.markdown("---")
        st.subheader("🔍 3. Giải thích Quyết định Mô hình (Explainable AI - XAI)")
        st.write("Trực quan hóa vùng ảnh phổi mà mô hình tập trung chú ý khi đưa ra chẩn đoán:")

        g1, g2, g3, g4 = st.columns(4)

        with g1:
            st.image(image_to_process, caption="1. Ảnh gốc (Original)", use_container_width=True)

        with g2:
            overlay_cnn = overlay_heatmap_on_image(image_to_process, xai["gradcam"], alpha=heatmap_alpha)
            st.image(overlay_cnn, caption="2. EfficientNet Grad-CAM", use_container_width=True)

        with g3:
            overlay_vit = overlay_heatmap_on_image(image_to_process, xai["vit_attention"], alpha=heatmap_alpha)
            st.image(overlay_vit, caption="3. ViT Attention Rollout", use_container_width=True)

        with g4:
            overlay_ens = overlay_heatmap_on_image(image_to_process, xai["ensemble_heatmap"], alpha=heatmap_alpha)
            st.image(overlay_ens, caption="4. Adaptive Fusion (Đề xuất)", use_container_width=True)


if __name__ == "__main__":
    main()
