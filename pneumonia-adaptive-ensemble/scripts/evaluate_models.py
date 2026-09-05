"""Comprehensive comparative evaluation script across standalone models and ensembles."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.config import load_config
from src.data.dataloader import build_dataloaders
from src.ensemble.adaptive_ensemble import AdaptiveEnsemble
from src.ensemble.fixed_ensemble import FixedEnsemble
from src.evaluation.evaluator import ModelEvaluator
from src.models.factory import build_model
from src.models.weights import load_model_weights
from src.utils.device import get_device
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models and ensembles on test split.")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset.yaml")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(42)
    logger = setup_logger(name="evaluate_models", log_filename="evaluate_models.log")
    device = get_device()
    dataset_cfg = load_config(args.dataset_config)

    logger.info("Building Test DataLoader...")
    loaders = build_dataloaders(
        data_dir=dataset_cfg.get("dataset", {}).get("raw_dir", "data/raw"),
        splits_dir=dataset_cfg.get("dataset", {}).get("splits_dir", "data/splits"),
        image_size=380,
        batch_size=16,
    )
    test_loader = loaders["test"]

    evaluator = ModelEvaluator(test_loader=test_loader, output_dir=args.results_dir, device=device, logger=logger)

    # 1. EfficientNet-B4
    logger.info("Evaluating [1/5] EfficientNet-B4 CNN Baseline...")
    cfg_cnn = load_config("configs/efficientnet_b4.yaml")
    model_cnn = build_model(cfg_cnn).to(device)
    if Path("models/efficientnet_b4/best.pt").exists():
        load_model_weights(model_cnn, "models/efficientnet_b4/best.pt", device=device)
    evaluator.evaluate_single_model(model_cnn, model_name="EfficientNet-B4", image_size=380)

    # 2. ViT-B/16
    logger.info("Evaluating [2/5] ViT-B/16 Transformer Baseline...")
    cfg_vit = load_config("configs/vit_b16.yaml")
    model_vit = build_model(cfg_vit).to(device)
    if Path("models/vit_b16/best.pt").exists():
        load_model_weights(model_vit, "models/vit_b16/best.pt", device=device)
    evaluator.evaluate_single_model(model_vit, model_name="ViT-B16", image_size=224)

    # 3. Fixed Ensemble
    logger.info("Evaluating [3/5] Fixed Ensemble...")
    fixed_ens = FixedEnsemble(model_cnn, model_vit, alpha=0.5).to(device)
    evaluator.evaluate_ensemble(fixed_ens, ensemble_name="Fixed-Ensemble")

    # 4. Adaptive Ensemble (Confidence-based)
    logger.info("Evaluating [4/5] Adaptive Ensemble (Confidence-based)...")
    adapt_conf = AdaptiveEnsemble(model_cnn, model_vit, strategy="confidence_based").to(device)
    evaluator.evaluate_ensemble(adapt_conf, ensemble_name="Adaptive-Ensemble-Confidence")

    # 5. Adaptive Ensemble (Hybrid)
    logger.info("Evaluating [5/5] Adaptive Ensemble (Hybrid Uncertainty + Confidence)...")
    adapt_hybrid = AdaptiveEnsemble(model_cnn, model_vit, strategy="hybrid").to(device)
    evaluator.evaluate_ensemble(adapt_hybrid, ensemble_name="Adaptive-Ensemble-Hybrid")

    # Generate Comparison ROC, PR curves, and McNemar significance tests
    logger.info("Generating multi-model comparison ROC curves, Reliability diagrams, and McNemar tests...")
    evaluator.generate_comparison_plots_and_tests()

    logger.info("Evaluation completed successfully! Results stored in results/")


if __name__ == "__main__":
    main()
