import numpy as np

class FixedEnsemble:
    def __init__(self, mode='soft', weights=None):
        self.mode = mode
        self.weights = weights
        
    def predict_probs(self, probs_list):
        probs_matrix = np.array(probs_list)
        if self.weights is not None:
            w = np.array(self.weights) / np.sum(self.weights)
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
            
        return ensemble_probs
