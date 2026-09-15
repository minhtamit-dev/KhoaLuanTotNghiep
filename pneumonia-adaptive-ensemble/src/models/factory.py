import torch
import torch.nn as nn
import timm

MODEL_ALIAS = {
    'efficientnet-b4': 'efficientnet_b4',
    'efficientnet_b4': 'efficientnet_b4',
    'vit-b/16': 'vit_base_patch16_224',
    'vit_b16': 'vit_base_patch16_224',
    'vit_base_patch16_224': 'vit_base_patch16_224',
    'resnet50': 'resnet50',
    'resnet-50': 'resnet50',
    'densenet121': 'densenet121',
    'densenet-121': 'densenet121',
    'convnext_base': 'convnext_base',
    'swin_base': 'swin_base_patch4_window7_224'
}

def build_model(model_name, num_classes=2, pretrained=True):
    name_clean = model_name.lower().strip()
    arch_name = MODEL_ALIAS.get(name_clean, name_clean)
    
    try:
        model = timm.create_model(arch_name, pretrained=pretrained, num_classes=num_classes)
        return model
    except Exception as e:
        raise ValueError(f"Mô hình '{model_name}' (timm architecture: '{arch_name}') không khởi tạo được: {e}")
