import torch
import torch.nn.functional as F
import numpy as np
import cv2

def generate_gradcam_heatmap(model, input_tensor, target_layer=None):
    model.eval()
    
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
                
    gradients = []
    activations = []
    
    def save_activation(module, input, output):
        activations.append(output)
        if output.requires_grad:
            output.register_hook(lambda grad: gradients.append(grad))
        
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
