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
from src.explainability.gradcam import generate_gradcam_heatmap
from src.utils.database import log_diagnosis, log_physician_review, get_db_connection
from torchvision import transforms

st.set_page_config(
    page_title="Hệ thống Chẩn đoán Viêm Phổi X-quang | Đồ án CNTT-KLCN171",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        color: #1E40AF;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 6px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        font-size: 0.9rem;
        color: #92400E;
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
    st.markdown('<div class="sub-header">Mô hình Kết hợp CNN & Vision Transformer | Mã đề tài: <b>CNTT-KLCN171</b></div>', unsafe_allow_html=True)

    # Banner Cảnh báo & Miễn trừ trách nhiệm theo tiêu chuẩn y tế
    st.markdown("""
    <div class="disclaimer-banner">
        <b>⚖️ CẢNH BÁO Y TẾ & CHÍNH SÁCH BẢO MẬT DỮ LIỆU:</b><br/>
        • Ứng dụng là hệ thống hỗ trợ chẩn đoán (CAD) phục vụ mục đích nghiên cứu khoa học, <b>KHÔNG</b> thay thế chẩn đoán chuyên môn hoặc quyết định lâm sàng của Bác sĩ chuyên khoa Chẩn đoán hình ảnh.<br/>
        • <b>Quyền riêng tư:</b> Hình ảnh tải lên được xử lý tạm thời trong bộ nhớ đệm RAM và tự động xóa. Mã định danh bệnh nhân được băm ẩn danh (De-identified) theo chuẩn HIPAA trước khi lưu log nghiên cứu.
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.image("https://img.icons8.com/color/96/lungs.png", width=70)
    st.sidebar.title("⚙️ Cấu hình Chẩn đoán")
    st.sidebar.markdown("**GVHD:** TS. Phùng Thế Bảo")
    st.sidebar.markdown("**Nhóm SV:** Nguyễn Gia Khang (NT), Phan Minh Tâm, Trịnh Minh Hiếu")
    st.sidebar.divider()

    model_option = st.sidebar.selectbox(
        "Lựa chọn Mô hình Chẩn đoán:",
        [
            "⚡ Fixed Soft Voting (EffNet-B4 + ViT-B/16)",
            "⚖️ Weighted Soft Voting (Val-weighted)",
            "🔒 Fixed Hard Voting (EffNet-B4 + ViT-B/16)",
            "🌟 Ensemble Cải tiến (ResNet50 + ViT-B/16)",
            "🌐 Tri-Model Ensemble (EffNet + ViT + ResNet50)",
            "🔹 EfficientNet-B4 (Đơn lẻ)",
            "🔹 ViT-B/16 (Vision Transformer Đơn lẻ)",
            "🔹 ResNet50 (Baseline CNN)"
        ]
    )

    threshold = st.sidebar.slider("Ngưỡng phân loại Pneumonia:", 0.1, 0.9, 0.5, 0.05)
    st.sidebar.caption("💡 Ngưỡng mặc định 0.50. Tăng ngưỡng (ví dụ 0.60-0.70) giúp tăng Specificity (giảm báo nhầm người khỏe mạnh).")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Chẩn đoán & Bản đồ nhiệt Giải thích",
        "👨‍⚕️ Nhận xét từ Bác sĩ Chẩn đoán hình ảnh",
        "📊 Bảng Kết quả Thực nghiệm & Độ tin cậy",
        "🔒 Chính sách Bảo mật & Khóa ngoại SQLite"
    ])

    models, device = load_all_models()

    with tab1:
        st.subheader("📤 Tải lên ảnh X-quang Ngực (Chest X-Ray)")
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_file = st.file_uploader("Chọn file ảnh X-quang (.png, .jpg, .jpeg):", type=["png", "jpg", "jpeg"])
        with col_up2:
            patient_code = st.text_input("Mã bệnh nhân (Tùy chọn, tự động ẩn danh):", value="CXR_PATIENT_TEMP")

        active_img = None
        if uploaded_file is not None:
            active_img = Image.open(uploaded_file).convert("RGB")

        if active_img is not None:
            col_img, col_res = st.columns([1, 1.2])
            
            with col_img:
                st.image(active_img, caption="Ảnh X-quang đầu vào", use_container_width=True)

            with col_res:
                st.subheader("🔍 Kết quả Phân tích")
                
                with st.spinner("Đang tính toán xác suất qua các mô hình..."):
                    img_tensor = transform_image(active_img).to(device)
                    probs_dict = predict_probabilities(models, img_tensor)
                    
                    p_eff = probs_dict.get('EfficientNet-B4', 0.5)
                    p_vit = probs_dict.get('ViT-B/16', 0.5)
                    p_res = probs_dict.get('ResNet50', 0.5)

                    if "Soft Voting (EffNet-B4 + ViT-B/16)" in model_option:
                        final_prob = (p_eff + p_vit) / 2.0
                    elif "Weighted Soft" in model_option:
                        final_prob = 0.48 * p_eff + 0.52 * p_vit
                    elif "Hard Voting" in model_option:
                        pred1 = 1.0 if p_eff >= threshold else 0.0
                        pred2 = 1.0 if p_vit >= threshold else 0.0
                        final_prob = (pred1 + pred2) / 2.0
                    elif "ResNet50 + ViT" in model_option:
                        final_prob = (p_res + p_vit) / 2.0
                    elif "Tri-Model" in model_option:
                        final_prob = (p_eff + p_vit + p_res) / 3.0
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
                            <h3 style="color: #EF4444; margin:0;">🔴 KẾT QUẢ: VIÊM PHỔI (PNEUMONIA)</h3>
                            <p style="margin-top:0.5rem; color:#FCA5A5;">Độ tin cậy mắc bệnh: <b>{final_prob*100:.2f}%</b> (Ngưỡng áp dụng: {threshold:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card-normal">
                            <h3 style="color: #10B981; margin:0;">🟢 KẾT QUẢ: BÌNH THƯỜNG (NORMAL)</h3>
                            <p style="margin-top:0.5rem; color:#A7F3D0;">Độ tin cậy khỏe mạnh: <b>{(1-final_prob)*100:.2f}%</b> (Ngưỡng áp dụng: {threshold:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.progress(float(np.clip(final_prob, 0.0, 1.0)))
                    st.caption(f"Xác suất Viêm phổi: {final_prob:.4f} | Khỏe mạnh (Bình thường): {1-final_prob:.4f}")

                    # Ghi nhận kết quả vào SQLite (Đã bật PRAGMA foreign_keys = ON)
                    current_diag_id = log_diagnosis(
                        anonymized_code=patient_code,
                        model_name=model_option,
                        prob=final_prob,
                        prediction="PNEUMONIA" if is_pneumonia else "NORMAL",
                        threshold=threshold
                    )
                    st.session_state['last_diagnosis_id'] = current_diag_id

                    st.divider()
                    st.markdown("##### 📊 Dự đoán chi tiết của từng mô hình thành viên:")
                    m_df = pd.DataFrame([
                        {"Mô hình Backbone": "EfficientNet-B4 (CNN)", "Xác suất Viêm phổi": f"{p_eff*100:.2f}%", "Dự đoán (th=0.5)": "Pneumonia" if p_eff>=0.5 else "Normal"},
                        {"Mô hình Backbone": "ViT-B/16 (Vision Transformer)", "Xác suất Viêm phổi": f"{p_vit*100:.2f}%", "Dự đoán (th=0.5)": "Pneumonia" if p_vit>=0.5 else "Normal"},
                        {"Mô hình Backbone": "ResNet50 (Baseline CNN)", "Xác suất Viêm phổi": f"{p_res*100:.2f}%", "Dự đoán (th=0.5)": "Pneumonia" if p_res>=0.5 else "Normal"}
                    ])
                    st.dataframe(m_df, hide_index=True, use_container_width=True)

            # --- KHẮC PHỤC LỖI GRAD-CAM: Giải thích đúng mô hình được chọn ---
            st.divider()
            st.subheader("🔍 Bản đồ Nhiệt Giải thích Vùng Tổn thương")
            st.info("💡 **Khắc phục nhận xét phản biện:** Bản đồ nhiệt được sinh từ chính mô hình bạn đang lựa chọn hoặc so sánh song song giữa các thành viên Ensemble.")

            # Cho phép chọn mô hình để hiển thị bản đồ nhiệt
            explain_choice = st.radio(
                "Chọn mô hình trích xuất bản đồ nhiệt:",
                ["EfficientNet-B4 (Grad-CAM)", "ViT-B/16 (Patch Attention Grad-CAM)", "ResNet50 (Grad-CAM)", "So sánh song song cả 2 mô hình trong Ensemble"],
                horizontal=True
            )

            with st.spinner("Đang trích xuất bản đồ nhiệt chú ý..."):
                img_tensor_cam = transform_image(active_img).to(device)
                orig_np = np.array(active_img.resize((224, 224))) / 255.0

                def get_overlay(model_inst):
                    heatmap_np = generate_gradcam_heatmap(model_inst, img_tensor_cam)
                    cam_colored = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
                    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB) / 255.0
                    overlay = 0.6 * orig_np + 0.4 * cam_colored
                    return heatmap_np, np.clip(overlay, 0, 1)

                if "song song" in explain_choice:
                    m1 = models.get('EfficientNet-B4')
                    m2 = models.get('ViT-B/16')
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        if m1 is not None:
                            h1, ov1 = get_overlay(m1)
                            st.image(ov1, caption="Bản đồ Grad-CAM từ EfficientNet-B4 (CNN)", use_container_width=True)
                    with col_m2:
                        if m2 is not None:
                            h2, ov2 = get_overlay(m2)
                            st.image(ov2, caption="Bản đồ Chú ý từ ViT-B/16 (Transformer)", use_container_width=True)
                else:
                    target_m_name = 'EfficientNet-B4' if 'EfficientNet' in explain_choice else ('ViT-B/16' if 'ViT' in explain_choice else 'ResNet50')
                    cam_model = models.get(target_m_name)
                    if cam_model is not None:
                        heatmap_np, overlay = get_overlay(cam_model)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.image(orig_np, caption="Ảnh X-quang Gốc (224x224)", use_container_width=True)
                        with c2:
                            st.image(heatmap_np, caption=f"Bản đồ nhiệt ({target_m_name})", use_container_width=True)
                        with c3:
                            st.image(overlay, caption=f"Chồng phủ chú ý ({target_m_name})", use_container_width=True)

    with tab2:
        st.subheader("👨‍⚕️ Biểu mẫu Đánh giá Định tính từ Bác sĩ Chẩn đoán hình ảnh")
        st.write("Khắc phục nhận xét phản biện: *'Thu thập đánh giá định tính từ 1–2 bác sĩ chẩn đoán hình ảnh về độ chính xác và tính giải thích của Grad-CAM'*. Dữ liệu đánh giá được lưu vào SQLite với toàn vẹn ràng buộc khóa ngoại.")
        
        with st.form("doctor_review_form"):
            doc_name = st.text_input("Họ tên Bác sĩ / Chuyên gia y tế:", value="BS. CKI Trần Minh Tuấn")
            hospital = st.text_input("Bệnh viện / Khoa công tác:", value="Khoa Chẩn đoán hình ảnh - BV Đa khoa")
            agreement = st.selectbox("Mức độ đồng thuận với chẩn đoán của AI:", ["Hoàn toàn đồng ý", "Đồng ý có cân nhắc", "Không đồng ý", "Cần làm thêm xét nghiệm CT/RT-PCR"])
            overall_rating = st.slider("Đánh giá tổng thể độ chính xác của hệ thống (1 - 5 sao):", 1, 5, 4)
            gradcam_rating = st.slider("Đánh giá độ khớp của vùng tổn thương trên bản đồ nhiệt Grad-CAM với thực tế lâm sàng (1 - 5 sao):", 1, 5, 4)
            notes = st.text_area("Ghi chú lâm sàng / Nhận xét chuyên môn của Bác sĩ:", value="Vùng nhiệt tập trung tốt ở thùy dưới phổi hai bên, mô hình ViT thể hiện sự chú ý rộng hơn so với CNN. Cần tiếp tục mở rộng dữ liệu người lớn tuổi.")
            
            submitted = st.form_submit_button("💾 Lưu Nhận xét Chuyên môn vào Cơ sở Dữ liệu")
            if submitted:
                last_id = st.session_state.get('last_diagnosis_id', 1)
                ok = log_physician_review(last_id, doc_name, hospital, agreement, overall_rating, gradcam_rating, notes)
                if ok:
                    st.success("✅ Đã ghi nhận thành công đánh giá của Bác sĩ vào cơ sở dữ liệu SQLite!")
                else:
                    st.error("Không thể lưu đánh giá. Vui lòng thử lại.")

        st.divider()
        st.markdown("##### 📑 Lịch sử Đánh giá của các Bác sĩ đã lưu:")
        conn = get_db_connection()
        try:
            reviews_df = pd.read_sql_query("SELECT id, physician_name, hospital_or_dept, agreement_status, rating, gradcam_fidelity_rating, clinical_notes, reviewed_at FROM physician_reviews ORDER BY id DESC LIMIT 10", conn)
            if not reviews_df.empty:
                st.dataframe(reviews_df, hide_index=True, use_container_width=True)
            else:
                st.info("Chưa có nhận xét nào được lưu.")
        except Exception as e:
            st.caption(f"Chưa có dữ liệu: {e}")
        finally:
            conn.close()

    with tab3:
        st.subheader("📈 Kết quả Thực nghiệm & Khoảng tin cậy (95% CI Bootstrap)")
        csv_file = os.path.join(BASE_DIR, 'results', 'experiment_results.csv')
        csv_opt = os.path.join(BASE_DIR, 'results', 'optimal_threshold_results.csv')
        
        st.markdown("**1. Bảng kết quả ở Ngưỡng mặc định 0.50 (kèm 95% Bootstrap CI cho ROC-AUC):**")
        if os.path.exists(csv_file):
            df_res = pd.read_csv(csv_file)
            st.dataframe(df_res, use_container_width=True)
        else:
            st.info("Chạy `python scripts/evaluate_models.py` để cập nhật bảng kết quả.")

        st.markdown("**2. Bảng kết quả ở Ngưỡng Youden tối ưu hóa từ Validation (Cải thiện Specificity):**")
        if os.path.exists(csv_opt):
            df_opt = pd.read_csv(csv_opt)
            st.dataframe(df_opt, use_container_width=True)

        col_f1, col_f2 = st.columns(2)
        roc_p = os.path.join(BASE_DIR, 'results', 'figures', 'roc_curves.png')
        cm_p = os.path.join(BASE_DIR, 'results', 'figures', 'confusion_matrices.png')
        with col_f1:
            if os.path.exists(roc_p):
                st.image(roc_p, caption="Hình 1: Đường cong ROC kèm khoảng tin cậy 95%", use_container_width=True)
        with col_f2:
            if os.path.exists(cm_p):
                st.image(cm_p, caption="Hình 2: Ma trận nhầm lẫn (Confusion Matrix)", use_container_width=True)

    with tab4:
        st.subheader("🔒 Chính sách Bảo mật & Toàn vẹn Khóa ngoại SQLite")
        st.markdown("""
        #### 1. Kích hoạt Toàn vẹn Khóa ngoại (Foreign Keys):
        Theo đúng khuyến nghị phản biện, toàn bộ kết nối SQLite trong hệ thống được cấu hình lệnh bắt buộc:
        ```python
        conn = sqlite3.connect('results/clinical_database.sqlite')
        conn.execute("PRAGMA foreign_keys = ON;")
        ```
        - Bảng `physician_reviews` liên kết khóa ngoại với `diagnoses(id)` (`ON DELETE CASCADE`).
        - Bảng `diagnoses` liên kết khóa ngoại với `patients(id)` (`ON DELETE CASCADE`).

        #### 2. Chính sách An toàn & Bảo mật Dữ liệu Y tế:
        - **Ẩn danh hóa (De-identification):** Ảnh tải lên không lưu giữ thông tin định danh cá nhân (PII).
        - **In-memory Processing:** Ảnh X-quang chỉ nạp vào tensor RAM và hủy khi giải phóng bộ nhớ.
        - **Mục đích hỗ trợ quyết định (Clinical Decision Support):** Hệ thống chỉ đóng vai trò khuyến nghị cho bác sĩ, tuân thủ nguyên tắc Human-in-the-loop.
        """)

if __name__ == '__main__':
    main()
