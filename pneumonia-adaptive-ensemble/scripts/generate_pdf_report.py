import os
import sys
import subprocess
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def generate_report():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docx_path = os.path.join(base_dir, 'CNTT-KLCN171_NguyenGiaKhang.docx')
    pdf_path = os.path.join(base_dir, 'CNTT-KLCN171_NguyenGiaKhang.pdf')
    root_pdf_path = os.path.join(os.path.dirname(base_dir), 'CNTT-KLCN171_NguyenGiaKhang.pdf')
    
    doc = Document()
    
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM\nKHOA CÔNG NGHỆ THÔNG TIN\n\n")
    run.bold = True
    run.font.size = Pt(13)
    
    run_title = p.add_run("BÁO CÁO KẾT QUẢ THỰC NGHIỆM ĐỒ ÁN TỐT NGHIỆP\n")
    run_title.bold = True
    run_title.font.size = Pt(16)
    run_title.font.color.rgb = RGBColor(0, 51, 102)
    
    run_sub = p.add_run("Đề tài: Nghiên cứu phương pháp ensemble cố định và thích ứng giữa CNN và Vision Transformer trong nhận diện viêm phổi từ ảnh X-quang ngực\n")
    run_sub.italic = True
    run_sub.font.size = Pt(12)
    
    p_meta = doc.add_paragraph()
    p_meta.add_run("Mã đề tài: CNTT-KLCN171\n").bold = True
    p_meta.add_run("Giảng viên hướng dẫn: TS. Phùng Thế Bảo\n").bold = True
    p_meta.add_run("Nhóm thực hiện: Nguyễn Gia Khang (Nhóm trưởng - MSSV: 2001230377), Phan Minh Tâm (MSSV: 2001230785), Trịnh Minh Hiếu (MSSV: 2001230245)\n\n")
    
    doc.add_heading("1. Tổng quan thực nghiệm", level=1)
    doc.add_paragraph(
        "Thực nghiệm được tiến hành trên bộ dữ liệu ảnh X-quang ngực công khai Chest X-Ray Images (Pneumonia) từ Kaggle "
        "với 5,216 ảnh huấn luyện và 624 ảnh kiểm thử độc lập (234 ảnh Bình thường và 390 ảnh Viêm phổi). "
        "Hệ thống so sánh hiệu năng giữa các mô hình học sâu đơn lẻ (EfficientNet-B4, ViT-B/16, ResNet50) "
        "và các phương pháp kết hợp mô hình Ensemble (Fixed Soft Voting, Hard Voting, Adaptive Weighting)."
    )
    
    doc.add_heading("2. Bảng tổng hợp kết quả đánh giá", level=1)
    
    results = [
        ["EfficientNet-B4", "85.90%", "83.26%", "96.92%", "67.52%", "0.8957", "0.9401"],
        ["ViT-B/16", "88.14%", "84.20%", "99.74%", "68.80%", "0.9131", "0.9799"],
        ["ResNet50", "91.99%", "89.91%", "98.21%", "81.62%", "0.9387", "0.9708"],
        ["Fixed Soft Voting", "87.66%", "83.95%", "99.23%", "68.38%", "0.9095", "0.9710"],
        ["Fixed Hard Voting", "90.22%", "88.34%", "97.18%", "78.63%", "0.9255", "0.8791"],
        ["Adaptive Weighting", "87.66%", "83.95%", "99.23%", "68.38%", "0.9095", "0.9713"]
    ]
    
    table = doc.add_table(rows=1, cols=7)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    headers = ["Mô hình / Method", "Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        hdr_cells[i].paragraphs[0].runs[0].bold = True
        
    for row_data in results:
        row_cells = table.add_row().cells
        for i, val in enumerate(row_data):
            row_cells[i].text = val
            
    doc.add_paragraph("\n")
    
    doc.add_heading("3. Biểu đồ đánh giá & Giải thích Grad-CAM", level=1)
    
    roc_img = os.path.join(base_dir, 'results', 'figures', 'roc_curves.png')
    cm_img = os.path.join(base_dir, 'results', 'figures', 'confusion_matrices.png')
    gradcam_img = os.path.join(base_dir, 'results', 'figures', 'gradcam', 'gradcam_pneumonia.png')
    
    if os.path.exists(roc_img):
        doc.add_paragraph("Hình 1: Đường cong ROC so sánh các mô hình và phương pháp Ensemble")
        doc.add_picture(roc_img, width=Inches(5.5))
        
    if os.path.exists(cm_img):
        doc.add_paragraph("Hình 2: Ma trận nhầm lẫn (Confusion Matrix) trên tập kiểm thử 624 ảnh")
        doc.add_picture(cm_img, width=Inches(5.5))
        
    if os.path.exists(gradcam_img):
        doc.add_paragraph("Hình 3: Bản đồ nhiệt Grad-CAM trực quan hóa vùng tổn thương viêm phổi")
        doc.add_picture(gradcam_img, width=Inches(5.5))
        
    doc.save(docx_path)
    print(f"[+] Saved Word report to {docx_path}")
    
    libreoffice_cmd = "/home/bonkerzz/.config/FlyEnv/env/composer/bin/libreoffice"
    if not os.path.exists(libreoffice_cmd):
        libreoffice_cmd = "libreoffice"
        
    cmd = [libreoffice_cmd, "--headless", "--convert-to", "pdf", docx_path, "--outdir", base_dir]
    subprocess.run(cmd, check=True)
    print(f"[+] Generated PDF report at: {pdf_path}")
    
    if os.path.exists(pdf_path):
        import shutil
        shutil.copy(pdf_path, root_pdf_path)
        print(f"[+] Copied PDF report to workspace root: {root_pdf_path}")

if __name__ == '__main__':
    generate_report()
