import numpy as np

class FixedEnsemble:
    """
    Module kết hợp mô hình Ensemble cố định.
    Khắc phục nhận xét phản biện:
    - Hỗ trợ Weighted Soft Voting theo hiệu năng Validation (thay vì chỉ chia đều 50/50).
    - Hỗ trợ Temperature Scaling để hiệu chuẩn độ tin cậy (calibration) trước khi lấy trung bình.
    - Hỗ trợ linh hoạt cặp [EffNet + ViT], [ResNet50 + ViT], và [EffNet + ViT + ResNet50].
    """
    def __init__(self, mode='soft', weights=None, temperatures=None):
        self.mode = mode
        self.weights = weights
        self.temperatures = temperatures
        
    def _apply_temperature_scaling(self, probs, temp=1.0):
        if temp == 1.0 or temp <= 0:
            return probs
        # Chuyển probs -> logits -> scaling -> softmax
        p = np.clip(probs, 1e-7, 1.0 - 1e-7)
        logits = np.log(p / (1.0 - p))
        scaled_logits = logits / temp
        scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        return scaled_probs

    def predict_probs(self, probs_list):
        processed_probs = []
        for i, p in enumerate(probs_list):
            t = 1.0
            if self.temperatures is not None and i < len(self.temperatures):
                t = self.temperatures[i]
            scaled = self._apply_temperature_scaling(np.array(p), temp=t)
            processed_probs.append(scaled)
            
        probs_matrix = np.array(processed_probs)
        
        if self.weights is not None:
            w = np.array(self.weights, dtype=float)
            w = w / np.sum(w)
            w = w[:, None]
        else:
            w = 1.0 / len(probs_list)
            
        if self.mode == 'soft':
            ensemble_probs = np.sum(probs_matrix * w, axis=0)
        elif self.mode == 'hard':
            preds_matrix = (probs_matrix >= 0.5).astype(float)
            ensemble_probs = np.sum(preds_matrix * w, axis=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        return np.clip(ensemble_probs, 0.0, 1.0)
