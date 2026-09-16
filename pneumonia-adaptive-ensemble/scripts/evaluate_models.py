import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data.dataset import PneumoniaDataset
from src.models.factory import build_model
from src.ensemble.fixed_ensemble import FixedEnsemble
from src.ensemble.adaptive_ensemble import AdaptiveEnsemble
from src.evaluation.metrics import evaluate_predictions

def evaluate_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    data_dir = os.path.join(BASE_DIR, 'data', 'chest_xray')
    test_dataset = PneumoniaDataset(data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    y_true = []
    for _, label in test_dataset:
        y_true.append(label)
    y_true = np.array(y_true)
    
    print(f"[+] Loaded Test Dataset: {len(test_dataset)} images (Normal: {np.sum(y_true==0)}, Pneumonia: {np.sum(y_true==1)})")
    
    models_dict = {
        'EfficientNet-B4': 'models/efficientnet_b4/best.pt',
        'ViT-B/16': 'models/vit_b16/best.pt',
        'ResNet50': 'models/resnet50/best.pt'
    }
    
    model_probs = {}
    for name, path in models_dict.items():
        full_path = os.path.join(BASE_DIR, path)
        print(f"[+] Evaluating {name}...")
        
        try:
            if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                model = build_model(name, num_classes=2, pretrained=False).to(device)
                model.load_state_dict(torch.load(full_path, map_location=device, weights_only=True))
            else:
                model = build_model(name, num_classes=2, pretrained=True).to(device)
        except Exception as e:
            print(f"[!] Warning building {name}: {e}. Trying pretrained fallback...")
            model = build_model(name, num_classes=2, pretrained=True).to(device)
            
        model.eval()
        
        probs_list = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                probs_list.extend(probs.cpu().numpy())
        model_probs[name] = np.array(probs_list)
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            
    fixed_soft = FixedEnsemble(mode='soft')
    fixed_hard = FixedEnsemble(mode='hard')
    
    eff_vit_pair = [model_probs['EfficientNet-B4'], model_probs['ViT-B/16']]
    
    model_probs['Fixed Soft Voting'] = fixed_soft.predict_probs(eff_vit_pair)
    model_probs['Fixed Hard Voting'] = fixed_hard.predict_probs(eff_vit_pair)
    
    results = []
    for name, probs in model_probs.items():
        metrics = evaluate_predictions(y_true, probs)
        metrics['Model / Method'] = name
        results.append(metrics)
        
    df = pd.DataFrame(results)
    cols = ['Model / Method', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-score', 'ROC-AUC', 'TP', 'TN', 'FP', 'FN']
    df = df[cols]
    
    print("\n================================================================================")
    print("FINAL BENCHMARK SUMMARY TABLE")
    print("================================================================================")
    print(df.to_string(index=False))
    print("================================================================================")
    
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'experiment_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[+] Results saved to {csv_path}")
    
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save ROC curves plot
    plt.figure(figsize=(9, 7))
    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = evaluate_predictions(y_true, probs)['ROC-AUC']
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curves - Single Models vs Fixed Ensembles')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(figures_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved ROC curves plot to {roc_path}")
    
    # Save Confusion Matrices plot (5 models)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.flatten()
    for idx, (name, probs) in enumerate(model_probs.items()):
        cm = confusion_matrix(y_true, (probs >= 0.5).astype(int))
        heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_flat[idx], cbar=False,
                xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
        axes_flat[idx].set_title(name)
        axes_flat[idx].set_xlabel('Predicted')
        axes_flat[idx].set_ylabel('Actual')
    
    # Turn off unused 6th subplot
    if len(model_probs) < len(axes_flat):
        axes_flat[-1].axis('off')
        
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved confusion matrices plot to {cm_path}")
    
    print("[+] Evaluation completed successfully!")

if __name__ == '__main__':
    evaluate_all()
