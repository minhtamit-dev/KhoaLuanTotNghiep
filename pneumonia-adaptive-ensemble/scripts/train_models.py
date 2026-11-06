import os
import sys
import gc
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data.dataset import PneumoniaDataset
from src.models.factory import build_model
from src.evaluation.metrics import evaluate_predictions

MODEL_CONFIGS = {
    'efficientnet_b4': {
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'epochs': 10,
        'batch_size': 16
    },
    'resnet50': {
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 10,
        'batch_size': 16
    },
    'vit_b16': {
        'lr': 3e-5,          # LR nhỏ hơn chuyên biệt cho Vision Transformer
        'weight_decay': 5e-2,
        'epochs': 10,
        'batch_size': 16
    }
}

def train_model(model_name, epochs=None, batch_size=None, lr=None, use_class_weights=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clean_name = model_name.lower().replace('-', '_').replace('/', '_')
    cfg = MODEL_CONFIGS.get(clean_name, {'lr': 1e-4, 'weight_decay': 1e-2, 'epochs': 10, 'batch_size': 16})
    
    actual_epochs = epochs or cfg['epochs']
    actual_batch_size = batch_size or cfg['batch_size']
    actual_lr = lr or cfg['lr']
    
    print(f"\n==================================================")
    print(f"[+] Training {model_name} on device: {device}")
    print(f"[+] Config: LR={actual_lr}, Epochs={actual_epochs}, BatchSize={actual_batch_size}, WeightDecay={cfg['weight_decay']}")
    print(f"==================================================")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    data_dir = os.path.join(BASE_DIR, 'data', 'chest_xray')
    
    # Sử dụng dataset mới: Tách validation theo Patient ID (15% train), không rò rỉ bệnh nhân
    train_dataset = PneumoniaDataset(data_dir, split='train')
    val_dataset = PneumoniaDataset(data_dir, split='val')
    
    print(f"[+] Train dataset: {len(train_dataset)} samples")
    print(f"[+] Validation dataset (Patient-aware split): {len(val_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False, num_workers=2)
    
    model = build_model(model_name, num_classes=2, pretrained=True).to(device)
    
    # Class weights để giải quyết vấn đề Specificity thấp do mất cân bằng dữ liệu 3:1
    if use_class_weights:
        weights = train_dataset.get_class_weights().to(device)
        print(f"[+] Using Inverse Class Weights for Loss: Normal={weights[0]:.2f}, Pneumonia={weights[1]:.2f}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=actual_epochs, eta_min=actual_lr * 0.05)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    best_f1 = 0.0
    save_dir = os.path.join(BASE_DIR, 'models', clean_name)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best.pt')
    
    for epoch in range(1, actual_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            
        scheduler.step()
        train_loss = train_loss / max(1, len(train_dataset))
        
        # Validation Evaluation
        model.eval()
        val_targets = []
        val_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                val_targets.extend(labels.numpy())
                val_probs.extend(probs.cpu().numpy())
                
        metrics = evaluate_predictions(val_targets, val_probs)
        f1 = metrics['F1-score']
        print(f"Epoch {epoch}/{actual_epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {train_loss:.4f} | Val Acc: {metrics['Accuracy']:.4f} | Val Spec: {metrics['Specificity']:.4f} | Val F1: {f1:.4f}")
        
        if f1 >= best_f1:
            best_f1 = f1
            if os.path.exists(best_path):
                os.remove(best_path)
            torch.save(model.state_dict(), best_path)
            print(f"    --> Saved best model checkpoint to {best_path} (F1: {best_f1:.4f})")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        
    return best_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Chest X-Ray Pneumonia Models")
    parser.add_argument('--model', type=str, default='all', help="Model to train: efficientnet_b4, vit_b16, resnet50, or all")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()
    
    if args.model == 'all':
        models_to_train = ['efficientnet_b4', 'vit_b16', 'resnet50']
    else:
        models_to_train = [args.model]
        
    for m in models_to_train:
        train_model(m, epochs=args.epochs)
