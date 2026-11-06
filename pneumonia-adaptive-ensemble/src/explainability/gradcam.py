import torch
import torch.nn.functional as F
import numpy as np
import cv2

def generate_gradcam_heatmap(model, input_tensor, target_layer=None):
    """
    Sinh bản đồ nhiệt Grad-CAM hỗ trợ cả CNN (ResNet, EfficientNet) và Vision Transformer (ViT).
    Khắc phục nhận xét:
    'Grad-CAM được sinh từ ResNet50, mô hình không nằm trong ensemble.
     ViT chưa có phương pháp giải thích.'
    """
    model.eval()
    is_vit = hasattr(model, 'blocks') and hasattr(model, 'patch_embed')
    
    # Tự động phát hiện target_layer phù hợp với từng kiến trúc
    if target_layer is None:
        if is_vit:
            # Đối với ViT: Hook vào norm1 của khối transformer cuối cùng
            target_layer = model.blocks[-1].norm1
        elif hasattr(model, 'conv_head') and isinstance(model.conv_head, torch.nn.Conv2d):
            # EfficientNet
            target_layer = model.conv_head
        elif hasattr(model, 'layer4'):
            # ResNet
            target_layer = model.layer4[-1]
        else:
            # Fallback duyệt ngược tìm conv cuối cùng
            for _, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break

    gradients = []
    activations = []
    
    def save_activation(module, inp, out):
        activations.append(out)
        if out.requires_grad:
            out.register_hook(lambda grad: gradients.append(grad))
        
    handle = None
    if target_layer is not None:
        handle = target_layer.register_forward_hook(save_activation)
    
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    score = output[0, pred_class]
    
    model.zero_grad()
    score.backward()
    
    if handle:
        handle.remove()
    
    if target_layer is not None and activations:
        act = activations[0].detach()
        grad = gradients[0].detach() if gradients else torch.ones_like(act)
        
        if is_vit:
            # Với ViT, loại bỏ CLS token ở index 0, reshape 196 patch tokens -> 14x14
            B = act.shape[0]
            C = act.shape[2] # 768
            # act: [B, 197, 768], grad: [B, 197, 768]
            patch_act = act[:, 1:, :].reshape(B, 14, 14, C).permute(0, 3, 1, 2) # [B, C, 14, 14]
            patch_grad = grad[:, 1:, :].reshape(B, 14, 14, C).permute(0, 3, 1, 2)
            
            weights = torch.mean(patch_grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * patch_act, dim=1, keepdim=True)
        else:
            # Standard CNN Grad-CAM
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.squeeze().cpu().numpy()
        cam_resized = cv2.resize(cam_np, (224, 224))
        return cam_resized
    else:
        return np.ones((224, 224), dtype=np.float32) * 0.5
