import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def compute_auc_ci(y_true, y_probs, n_bootstraps=1000, alpha=0.05, seed=42):
    """
    Tính khoảng tin cậy 95% cho ROC-AUC bằng phương pháp Non-parametric Bootstrap.
    Khắc phục nhận xét: 'AUC không có khoảng tin cậy (DeLong hoặc bootstrap)'.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_probs[indices])
        bootstrapped_scores.append(score)
        
    if not bootstrapped_scores:
        base_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
        return float(base_auc), float(base_auc)
        
    lower = np.percentile(bootstrapped_scores, (alpha / 2.0) * 100)
    upper = np.percentile(bootstrapped_scores, (1.0 - alpha / 2.0) * 100)
    return float(lower), float(upper)

def find_optimal_threshold(y_true, y_probs, method='youden', target_sensitivity=0.95):
    """
    Tìm ngưỡng phân loại tối ưu trên tập Validation (khắc phục lỗi chọn ngưỡng trên tập test).
    - 'youden': Tối đa hóa chỉ số Youden's J = Sensitivity + Specificity - 1 (tối ưu hóa độ đặc hiệu).
    - 'f1': Tối đa hóa F1-Score.
    - 'target_sensitivity': Chọn ngưỡng cao nhất mà vẫn giữ Sensitivity >= target_sensitivity.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    if method == 'youden':
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
    elif method == 'f1':
        candidate_thresholds = np.linspace(0.1, 0.9, 81)
        best_f1 = -1.0
        best_threshold = 0.5
        for th in candidate_thresholds:
            preds = (y_probs >= th).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = th
    elif method == 'target_sensitivity':
        valid_indices = np.where(tpr >= target_sensitivity)[0]
        if len(valid_indices) > 0:
            # Chọn ngưỡng lớn nhất (để tối đa Specificity) mà TPR >= target
            best_idx = valid_indices[-1]
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = 0.5
    else:
        best_threshold = 0.5
        
    return float(np.clip(best_threshold, 0.05, 0.95))

def evaluate_predictions(y_true, y_probs, threshold=0.5, compute_ci=False):
    """
    Đánh giá toàn diện các chỉ số phân loại y khoa với ngưỡng xác định.
    Bao gồm: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score, ROC-AUC, 95% CI.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs)
    else:
        auc = 0.5
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    res = {
        'Threshold': float(threshold),
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall (Sensitivity)': float(rec),
        'Specificity': float(spec),
        'F1-score': float(f1),
        'ROC-AUC': float(auc),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }
    
    if compute_ci and len(np.unique(y_true)) > 1:
        ci_low, ci_high = compute_auc_ci(y_true, y_probs)
        res['AUC_CI_Lower'] = ci_low
        res['AUC_CI_Upper'] = ci_high
        res['ROC-AUC (95% CI)'] = f"{auc:.4f} ({ci_low:.4f} - {ci_high:.4f})"
        
    return res
