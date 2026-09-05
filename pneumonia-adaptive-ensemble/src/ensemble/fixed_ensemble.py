"""Fixed Ensemble combining CNN and ViT with constant weight alpha."""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class FixedEnsemble(nn.Module):
    """Fixed-weight Weighted Average Ensemble of EfficientNet-B4 and ViT-B/16."""

    def __init__(
        self,
        model_cnn: nn.Module,
        model_vit: nn.Module,
        alpha: float = 0.5,
    ):
        """Initialize FixedEnsemble.

        Args:
            model_cnn: EfficientNet-B4 PyTorch model.
            model_vit: ViT-B/16 PyTorch model.
            alpha: Weight for the CNN model (1 - alpha for ViT).
        """
        super().__init__()
        self.model_cnn = model_cnn
        self.model_vit = model_vit
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def set_alpha(self, alpha: float) -> None:
        """Update alpha weighting factor."""
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def forward(
        self,
        x_cnn: torch.Tensor,
        x_vit: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass for Fixed Ensemble.

        Args:
            x_cnn: Image tensor resized for CNN (380x380).
            x_vit: Image tensor resized for ViT (224x224). If None, will resize x_cnn.

        Returns:
            Tuple of (fused_probabilities, info_dict)
        """
        if x_vit is None:
            x_vit = torch.nn.functional.interpolate(
                x_cnn, size=(224, 224), mode="bicubic", align_corners=False
            )

        probs_cnn = self.model_cnn.predict_proba(x_cnn)
        probs_vit = self.model_vit.predict_proba(x_vit)

        fused_probs = self.alpha * probs_cnn + (1.0 - self.alpha) * probs_vit

        info = {
            "probs_cnn": probs_cnn,
            "probs_vit": probs_vit,
            "weight_cnn": torch.full((x_cnn.size(0), 1), self.alpha, device=x_cnn.device),
            "weight_vit": torch.full((x_cnn.size(0), 1), 1.0 - self.alpha, device=x_cnn.device),
            "strategy": "fixed_ensemble",
        }

        return fused_probs, info

    @torch.no_grad()
    def optimize_alpha(
        self,
        val_loader: Any,
        device: torch.device,
        search_range: Tuple[float, float] = (0.0, 1.0),
        step: float = 0.05,
    ) -> Tuple[float, float]:
        """Grid-search the optimal alpha value on the validation dataset.

        Args:
            val_loader: DataLoader for validation split.
            device: Computing device.
            search_range: Min and max alpha.
            step: Step size for search.

        Returns:
            Tuple of (best_alpha, best_f1_score).
        """
        self.model_cnn.eval()
        self.model_vit.eval()

        all_probs_cnn = []
        all_probs_vit = []
        all_targets = []

        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            images_vit = torch.nn.functional.interpolate(
                images, size=(224, 224), mode="bicubic", align_corners=False
            )

            probs_cnn = self.model_cnn.predict_proba(images)
            probs_vit = self.model_vit.predict_proba(images_vit)

            all_probs_cnn.append(probs_cnn.cpu())
            all_probs_vit.append(probs_vit.cpu())
            all_targets.append(targets.cpu())

        cat_probs_cnn = torch.cat(all_probs_cnn, dim=0).numpy()
        cat_probs_vit = torch.cat(all_probs_vit, dim=0).numpy()
        cat_targets = torch.cat(all_targets, dim=0).numpy()

        alphas = np.arange(search_range[0], search_range[1] + 1e-5, step)
        best_alpha = 0.5
        best_f1 = -1.0

        for a in alphas:
            fused = a * cat_probs_cnn + (1.0 - a) * cat_probs_vit
            preds = np.argmax(fused, axis=1)
            score = f1_score(cat_targets, preds, average="binary", zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_alpha = float(a)

        self.alpha = best_alpha
        return best_alpha, best_f1
