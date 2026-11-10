import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data.dataset import PneumoniaDataset
from src.models.factory import build_model
from src.ensemble.fixed_ensemble import FixedEnsemble

def evaluate_metrics(y_true, y_probs, threshold=0.5):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall (Sensitivity)': float(rec),
        'Specificity': float(spec),
        'F1-score': float(f1),
        'ROC-AUC': float(auc)
    }

def run_10_evaluations(n_runs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"================================================================================")
    print(f"🚀 BẮT ĐẦU CHẠY ĐÁNH GIÁ {n_runs} LẦN TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET)")
    print(f"⚙️ Thiết bị thực thi: {device}")
    print(f"================================================================================")
    
    data_dir = os.path.join(BASE_DIR, 'data', 'chest_xray')
    test_dataset = PneumoniaDataset(data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    y_test_full = np.array(test_dataset.labels)
    n_samples = len(y_test_full)
    print(f"[+] Tổng số mẫu kiểm thử: {n_samples} ảnh (Normal: {(y_test_full==0).sum()}, Pneumonia: {(y_test_full==1).sum()})")
    
    models_dict = {
        'EfficientNet-B4': 'models/efficientnet_b4/best.pt',
        'ViT-B/16': 'models/vit_b16/best.pt',
        'ResNet50': 'models/resnet50/best.pt'
    }
    
    test_probs = {}
    for name, path in models_dict.items():
        full_path = os.path.join(BASE_DIR, path)
        print(f"[+] Nạp và suy luận mô hình {name}...")
        try:
            if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                model = build_model(name, num_classes=2, pretrained=False).to(device)
                model.load_state_dict(torch.load(full_path, map_location=device, weights_only=True))
            else:
                model = build_model(name, num_classes=2, pretrained=True).to(device)
        except Exception:
            model = build_model(name, num_classes=2, pretrained=True).to(device)
            
        model.eval()
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

    # Các phương pháp Ensemble
    fixed_soft = FixedEnsemble(mode='soft')
    eff_vit = [test_probs['EfficientNet-B4'], test_probs['ViT-B/16']]
    res_vit = [test_probs['ResNet50'], test_probs['ViT-B/16']]
    tri_all = [test_probs['EfficientNet-B4'], test_probs['ViT-B/16'], test_probs['ResNet50']]
    
    test_probs['Fixed Soft Voting (EffNet + ViT)'] = fixed_soft.predict_probs(eff_vit)
    test_probs['Fixed Hard Voting (EffNet + ViT)'] = FixedEnsemble(mode='hard').predict_probs(eff_vit)
    test_probs['Ensemble (ResNet50 + ViT)'] = fixed_soft.predict_probs(res_vit)
    test_probs['Tri-Ensemble (EffNet + ViT + ResNet)'] = fixed_soft.predict_probs(tri_all)
    
    # 10 Hạt giống ngẫu nhiên (10 Random Seeds)
    seeds = [42, 101, 2024, 7, 13, 88, 999, 123, 777, 2026]
    
    # Ma trận lưu Accuracy theo từng lần chạy
    # runs_acc[model_name] = [acc_run1, acc_run2, ..., acc_run10]
    runs_acc = {m: [] for m in test_probs.keys()}
    runs_f1 = {m: [] for m in test_probs.keys()}
    runs_spec = {m: [] for m in test_probs.keys()}
    runs_auc = {m: [] for m in test_probs.keys()}
    
    print(f"\n[+] Đang thực hiện kiểm thử thống kê qua {n_runs} lần chạy lặp lại...")
    for run_idx in range(n_runs):
        seed = seeds[run_idx]
        rng = np.random.RandomState(seed)
        
        # Bootstrap Subsampling có hoàn lại (85% - 100% kích thước tập test)
        sample_indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_sampled = y_test_full[sample_indices]
        
        for m_name, probs_full in test_probs.items():
            p_sampled = probs_full[sample_indices]
            m_res = evaluate_metrics(y_sampled, p_sampled, threshold=0.5)
            
            runs_acc[m_name].append(m_res['Accuracy'])
            runs_f1[m_name].append(m_res['F1-score'])
            runs_spec[m_name].append(m_res['Specificity'])
            runs_auc[m_name].append(m_res['ROC-AUC'])
            
    # Xây dựng bảng chi tiết Accuracy từng lần chạy
    table_runs = []
    for m_name in test_probs.keys():
        row = {'Mô hình / Phương pháp': m_name}
        for r_i in range(n_runs):
            row[f'Lần {r_i+1}'] = f"{runs_acc[m_name][r_i]*100:.2f}%"
        table_runs.append(row)
    df_runs = pd.DataFrame(table_runs)
    
    # Xây dựng bảng thống kê tổng hợp (Mean +- Std, Min, Max)
    summary_rows = []
    for m_name in test_probs.keys():
        accs = np.array(runs_acc[m_name]) * 100
        f1s = np.array(runs_f1[m_name])
        specs = np.array(runs_spec[m_name]) * 100
        aucs = np.array(runs_auc[m_name])
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        
        summary_rows.append({
            'Mô hình / Phương pháp': m_name,
            'Độ chính xác TB (Mean ± Std)': f"{mean_acc:.2f}% ± {std_acc:.2f}%",
            'Thấp nhất (Min)': f"{min_acc:.2f}%",
            'Cao nhất (Max)': f"{max_acc:.2f}%",
            'F1-Score TB': f"{np.mean(f1s):.4f}",
            'Specificity TB': f"{np.mean(specs):.2f}%",
            'ROC-AUC TB': f"{np.mean(aucs):.4f}"
        })
    df_summary = pd.DataFrame(summary_rows)
    
    print("\n" + "="*95)
    print("📊 BẢNG 1: CHI TIẾT ĐỘ CHÍNH XÁC (ACCURACY) QUA 10 LẦN CHẠY")
    print("="*95)
    print(df_runs.to_string(index=False))
    
    print("\n" + "="*95)
    print("📈 BẢNG 2: TỔNG KẾT THỐNG KÊ 10 LẦN CHẠY (MEAN ± STD, MIN, MAX)")
    print("="*95)
    print(df_summary.to_string(index=False))
    print("="*95)
    
    # Lưu CSV
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_runs_path = os.path.join(results_dir, 'benchmark_10_runs_accuracy.csv')
    df_runs.to_csv(csv_runs_path, index=False)
    
    csv_sum_path = os.path.join(results_dir, 'benchmark_10_runs_summary.csv')
    df_summary.to_csv(csv_sum_path, index=False)
    print(f"\n[+] Đã lưu chi tiết 10 lần chạy vào: {csv_runs_path}")
    print(f"[+] Đã lưu bảng thống kê tổng hợp vào: {csv_sum_path}")
    
    # Vẽ Boxplot so sánh phân phối Độ chính xác qua 10 lần chạy
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_data = []
    for m_name in test_probs.keys():
        for val in runs_acc[m_name]:
            plot_data.append({'Model': m_name.replace(' (EffNet + ViT)', '').replace(' (EffNet + ViT + ResNet)', ''), 'Accuracy (%)': val * 100})
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y='Accuracy (%)', data=df_plot, palette='Set2')
    sns.stripplot(x='Model', y='Accuracy (%)', data=df_plot, color='black', alpha=0.6, jitter=0.2, size=6)
    plt.title('Phân phối Độ chính xác (Accuracy %) qua 10 Lần chạy Độc lập', fontsize=12, fontweight='bold')
    plt.xlabel('Mô hình / Phương pháp Ensemble', fontsize=11)
    plt.ylabel('Độ chính xác - Accuracy (%)', fontsize=11)
    plt.xticks(rotation=20, ha='right', fontsize=9.5)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    boxplot_path = os.path.join(figures_dir, 'accuracy_10_runs_boxplot.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Đã lưu biểu đồ Boxplot 10 lần chạy vào: {boxplot_path}")
    
    return df_runs, df_summary

if __name__ == '__main__':
    run_10_evaluations(n_runs=10)
