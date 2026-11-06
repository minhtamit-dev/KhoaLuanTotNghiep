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
from src.evaluation.metrics import evaluate_predictions, find_optimal_threshold, compute_auc_ci

def evaluate_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    data_dir = os.path.join(BASE_DIR, 'data', 'chest_xray')
    val_dataset = PneumoniaDataset(data_dir, split='val')
    test_dataset = PneumoniaDataset(data_dir, split='test')
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    y_test = np.array([label for _, label in test_dataset])
    y_val = np.array([label for _, label in val_dataset]) if len(val_dataset) > 0 else y_test
    
    print(f"[+] Loaded Val Dataset: {len(val_dataset)} images | Test Dataset: {len(test_dataset)} images")
    
    models_dict = {
        'EfficientNet-B4': 'models/efficientnet_b4/best.pt',
        'ViT-B/16': 'models/vit_b16/best.pt',
        'ResNet50': 'models/resnet50/best.pt'
    }
    
    val_probs = {}
    test_probs = {}
    
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
        
        # 1. Dự đoán trên Validation
        vp = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                vp.extend(probs.cpu().numpy())
        val_probs[name] = np.array(vp) if vp else np.full(len(y_val), 0.5)
        
        # 2. Dự đoán trên Test Set
        tp = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                tp.extend(probs.cpu().numpy())
        test_probs[name] = np.array(tp)
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    # --- Ensemble Combinations ---
    # Cặp gốc đề tài: EfficientNet-B4 + ViT-B/16
    eff_vit_test = [test_probs['EfficientNet-B4'], test_probs['ViT-B/16']]
    eff_vit_val = [val_probs['EfficientNet-B4'], val_probs['ViT-B/16']]
    
    # 1. Fixed Soft Voting (Equal Weights 50/50)
    fixed_soft = FixedEnsemble(mode='soft')
    test_probs['Fixed Soft Voting (EffNet + ViT)'] = fixed_soft.predict_probs(eff_vit_test)
    val_probs['Fixed Soft Voting (EffNet + ViT)'] = fixed_soft.predict_probs(eff_vit_val)
    
    # 2. Fixed Hard Voting
    fixed_hard = FixedEnsemble(mode='hard')
    test_probs['Fixed Hard Voting (EffNet + ViT)'] = fixed_hard.predict_probs(eff_vit_test)
    val_probs['Fixed Hard Voting (EffNet + ViT)'] = fixed_hard.predict_probs(eff_vit_val)
    
    # 3. Weighted Soft Voting (Trọng số theo AUC trên Validation)
    val_auc_eff = evaluate_predictions(y_val, val_probs['EfficientNet-B4'])['ROC-AUC']
    val_auc_vit = evaluate_predictions(y_val, val_probs['ViT-B/16'])['ROC-AUC']
    w_sum = val_auc_eff + val_auc_vit
    weights_val = [val_auc_eff / w_sum, val_auc_vit / w_sum]
    val_weighted_soft = FixedEnsemble(mode='soft', weights=weights_val)
    test_probs['Weighted Soft Voting (Val-weighted)'] = val_weighted_soft.predict_probs(eff_vit_test)
    val_probs['Weighted Soft Voting (Val-weighted)'] = val_weighted_soft.predict_probs(eff_vit_val)
    
    # 4. Biến thể gợi ý phản biện: Cặp ResNet50 + ViT-B/16
    res_vit_test = [test_probs['ResNet50'], test_probs['ViT-B/16']]
    res_vit_val = [val_probs['ResNet50'], val_probs['ViT-B/16']]
    test_probs['Ensemble (ResNet50 + ViT)'] = fixed_soft.predict_probs(res_vit_test)
    val_probs['Ensemble (ResNet50 + ViT)'] = fixed_soft.predict_probs(res_vit_val)
    
    # 5. Tri-Model Ensemble (EfficientNet + ViT + ResNet50)
    tri_test = [test_probs['EfficientNet-B4'], test_probs['ViT-B/16'], test_probs['ResNet50']]
    tri_val = [val_probs['EfficientNet-B4'], val_probs['ViT-B/16'], val_probs['ResNet50']]
    test_probs['Tri-Ensemble (EffNet + ViT + ResNet)'] = fixed_soft.predict_probs(tri_test)
    val_probs['Tri-Ensemble (EffNet + ViT + ResNet)'] = fixed_soft.predict_probs(tri_val)

    # --- Tính toán chỉ số & Ngưỡng Youden khách quan từ Validation ---
    results_th05 = []
    results_optimal = []
    
    for name in test_probs.keys():
        t_probs = test_probs[name]
        v_probs = val_probs[name]
        
        # Ngưỡng Youden tối ưu tìm trên tập validation
        opt_th = find_optimal_threshold(y_val, v_probs, method='youden')
        
        # Đánh giá trên test với threshold mặc định 0.5
        m05 = evaluate_predictions(y_test, t_probs, threshold=0.5, compute_ci=True)
        m05['Model / Method'] = name
        results_th05.append(m05)
        
        # Đánh giá trên test với ngưỡng tối ưu Youden từ Val
        m_opt = evaluate_predictions(y_test, t_probs, threshold=opt_th, compute_ci=True)
        m_opt['Model / Method'] = name
        m_opt['Val Optimal Threshold'] = f"{opt_th:.2f}"
        results_optimal.append(m_opt)

    df_th05 = pd.DataFrame(results_th05)
    cols = ['Model / Method', 'Threshold', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-score', 'ROC-AUC', 'ROC-AUC (95% CI)', 'TP', 'TN', 'FP', 'FN']
    df_th05 = df_th05[cols]
    
    df_opt = pd.DataFrame(results_optimal)
    cols_opt = ['Model / Method', 'Val Optimal Threshold', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-score', 'ROC-AUC', 'TP', 'TN', 'FP', 'FN']
    df_opt = df_opt[cols_opt]

    print("\n================================================================================")
    print("BENCHMARK SUMMARY TABLE AT DEFAULT THRESHOLD 0.50 (WITH 95% BOOTSTRAP CI)")
    print("================================================================================")
    print(df_th05.to_string(index=False))
    print("\n================================================================================")
    print("BENCHMARK SUMMARY TABLE AT VALIDATION-TUNED YOUDEN THRESHOLD (OPTIMIZED SPECIFICITY)")
    print("================================================================================")
    print(df_opt.to_string(index=False))
    print("================================================================================")
    
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'experiment_results.csv')
    df_th05.to_csv(csv_path, index=False)
    
    csv_opt_path = os.path.join(results_dir, 'optimal_threshold_results.csv')
    df_opt.to_csv(csv_opt_path, index=False)
    print(f"[+] Results saved to {csv_path} and {csv_opt_path}")
    
    # Save ROC curves plot
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    for name, probs in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        ci_low, ci_high = compute_auc_ci(y_test, probs)
        auc_val = evaluate_predictions(y_test, probs)['ROC-AUC']
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f} [95% CI: {ci_low:.3f}-{ci_high:.3f}])", linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Guess (AUC = 0.5000)")
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves - Single vs Ensembles', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8.5)
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(figures_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved ROC curves plot with 95% CI to {roc_path}")
    
    # Save Confusion Matrices plot
    n_models = len(test_probs)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for idx, (name, probs) in enumerate(list(test_probs.items())[:6]):
        cm = confusion_matrix(y_test, (probs >= 0.5).astype(int))
        heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_flat[idx], cbar=False,
                xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
        axes_flat[idx].set_title(name, fontsize=10, fontweight='bold')
        axes_flat[idx].set_xlabel('Predicted')
        axes_flat[idx].set_ylabel('Actual')
        
    if n_models < len(axes_flat):
        for rem in range(n_models, len(axes_flat)):
            axes_flat[rem].axis('off')
            
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved confusion matrices plot to {cm_path}")
    print("[+] Evaluation completed successfully!")

if __name__ == '__main__':
    evaluate_all()
