import numpy as np

class AdaptiveEnsemble:
    def __init__(self, strategy='confidence'):
        self.strategy = strategy
        
    def predict_probs(self, probs_list):
        probs_matrix = np.array(probs_list)
        
        if self.strategy == 'confidence':
            confidences = np.abs(probs_matrix - 0.5) * 2.0 + 1e-6
            weights = confidences / np.sum(confidences, axis=0, keepdims=True)
            ensemble_probs = np.sum(probs_matrix * weights, axis=0)
        elif self.strategy == 'entropy':
            p = np.clip(probs_matrix, 1e-6, 1 - 1e-6)
            entropies = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
            inv_entropies = 1.0 / (entropies + 1e-6)
            weights = inv_entropies / np.sum(inv_entropies, axis=0, keepdims=True)
            ensemble_probs = np.sum(probs_matrix * weights, axis=0)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        return ensemble_probs
