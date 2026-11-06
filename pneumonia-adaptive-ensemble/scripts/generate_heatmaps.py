import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data.dataset import PneumoniaDataset
from src.models.factory import build_model
from src.explainability.gradcam import generate_gradcam_heatmap

def generate_heatmaps():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Generating Grad-CAM and ViT Attention heatmaps on {device}...")
    
    data_dir = os.path.join(BASE_DIR, 'data', 'chest_xray')
    test_dataset = PneumoniaDataset(data_dir, split='test')
    
    save_dir = os.path.join(BASE_DIR, 'results', 'figures', 'gradcam')
    os.makedirs(save_dir, exist_ok=True)
    
    # Tìm mẫu Normal và Pneumonia đại diện trong test set
    normal_idx = None
    pneumonia_idx = None
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        if label == 0 and normal_idx is None:
            normal_idx = idx
        elif label == 1 and pneumonia_idx is None:
            pneumonia_idx = idx
        if normal_idx is not None and pneumonia_idx is not None:
            break
            
    # Danh sách các mô hình cần sinh bản đồ nhiệt giải thích quyết định
    models_to_explain = [
        ('EfficientNet-B4', 'models/efficientnet_b4/best.pt', 'effnet'),
        ('ViT-B/16', 'models/vit_b16/best.pt', 'vit'),
        ('ResNet50', 'models/resnet50/best.pt', 'resnet')
    ]
    
    for model_name, ckpt_rel, tag_prefix in models_to_explain:
        full_ckpt = os.path.join(BASE_DIR, ckpt_rel)
        print(f"[+] Generating heatmaps for model: {model_name}...")
        
        try:
            if os.path.exists(full_ckpt) and os.path.getsize(full_ckpt) > 0:
                model = build_model(model_name, num_classes=2, pretrained=False).to(device)
                model.load_state_dict(torch.load(full_ckpt, map_location=device, weights_only=True))
            else:
                model = build_model(model_name, num_classes=2, pretrained=True).to(device)
        except Exception:
            model = build_model(model_name, num_classes=2, pretrained=True).to(device)
            
        model.eval()
        
        for sample_type, idx in [('normal', normal_idx), ('pneumonia', pneumonia_idx)]:
            if idx is None:
                continue
            img_tensor, label = test_dataset[idx]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            heatmap_np = generate_gradcam_heatmap(model, input_tensor)
            
            orig_img = img_tensor.numpy().transpose(1, 2, 0)
            orig_img = (orig_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            orig_img = np.clip(orig_img, 0, 1)
            
            cam_colored = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
            cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB) / 255.0
            overlay = 0.6 * orig_img + 0.4 * cam_colored
            overlay = np.clip(overlay, 0, 1)
            
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            axes[0].imshow(orig_img)
            axes[0].set_title("Ảnh X-quang Gốc (224x224)", fontsize=10)
            axes[0].axis('off')
            
            axes[1].imshow(heatmap_np, cmap='jet')
            method_desc = "ViT Attention Map" if "ViT" in model_name else "Grad-CAM Heatmap"
            axes[1].set_title(f"{method_desc} ({model_name})", fontsize=10, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay Vùng Chú Ý ({'Viêm phổi' if label==1 else 'Bình thường'})", fontsize=10)
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Lưu file đặc thù từng model
            out_filename = f"gradcam_{tag_prefix}_{sample_type}.png"
            out_path = os.path.join(save_dir, out_filename)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Đồng thời nếu là pneumonia của EfficientNet-B4 hoặc ResNet50, cập nhật file mặc định cho báo cáo PDF
            if tag_prefix == 'effnet':
                rep_path = os.path.join(save_dir, f"gradcam_{sample_type}.png")
                cv2.imwrite(rep_path, cv2.cvtColor(np.uint8(255 * overlay), cv2.COLOR_RGB2BGR))
                
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    print(f"[+] All explainability heatmaps successfully generated in: {save_dir}")

if __name__ == '__main__':
    generate_heatmaps()
