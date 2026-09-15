import os
import sys
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data.dataset import PneumoniaDataset
from src.models.factory import build_model
from src.evaluation.metrics import evaluate_predictions

def train_model(model_name, epochs=3, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==================================================")
    print(f"[+] Training {model_name} on device: {device}")
    print(f"==================================================")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    data_dir = os.path.join(BASE_DIR, 'data', 'chest_xray')
    train_dataset = PneumoniaDataset(data_dir, split='train')
    val_dataset = PneumoniaDataset(data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = build_model(model_name, num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    best_f1 = 0.0
    save_dir = os.path.join(BASE_DIR, 'models', model_name.lower().replace('-', '_').replace('/', '_'))
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best.pt')
    
    for epoch in range(1, epochs + 1):
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
            
        train_loss = train_loss / len(train_dataset)
        
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
        print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {metrics['Accuracy']:.4f} | Val F1: {f1:.4f}")
        
        if f1 >= best_f1:
            best_f1 = f1
            if os.path.exists(best_path):
                os.remove(best_path)
            torch.save(model.state_dict(), best_path)
            print(f"    --> Saved best model checkpoint to {best_path} ({os.path.getsize(best_path)} bytes)")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        
    return best_path

if __name__ == '__main__':
    for name in ['efficientnet_b4', 'vit_b16', 'resnet50']:
        train_model(name, epochs=3, batch_size=16)
