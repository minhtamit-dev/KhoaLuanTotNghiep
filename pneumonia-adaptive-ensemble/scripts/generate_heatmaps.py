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
    print(f"[+] Generating Grad-CAM heatmaps on {device}...")
    
    data_dir = os.path.join(BASE_DIR, 'data', 'chest_xray')
    test_dataset = PneumoniaDataset(data_dir, split='test')
    
    model_name = 'ResNet50'
    model_path = os.path.join(BASE_DIR, 'models', 'resnet50', 'best.pt')
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        model_path = os.path.join(BASE_DIR, 'models', 'efficientnet_b4', 'best.pt')
        model_name = 'EfficientNet-B4'
        
    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            model = build_model(model_name, num_classes=2, pretrained=False).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
            model = build_model(model_name, num_classes=2, pretrained=True).to(device)
    except Exception as e:
        print(f"[!] Warning loading Grad-CAM model: {e}. Falling back to pretrained model.")
        model = build_model(model_name, num_classes=2, pretrained=True).to(device)
        
    model.eval()
    
    save_dir = os.path.join(BASE_DIR, 'results', 'figures', 'gradcam')
    os.makedirs(save_dir, exist_ok=True)
    
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
            
    for tag, idx in [('gradcam_normal.png', normal_idx), ('gradcam_pneumonia.png', pneumonia_idx)]:
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
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_img)
        axes[0].set_title("Original X-Ray Image")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_np, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (Class: {'Pneumonia' if label==1 else 'Normal'})")
        axes[2].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(save_dir, tag)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved Grad-CAM visualization to {out_path}")

if __name__ == '__main__':
    generate_heatmaps()
