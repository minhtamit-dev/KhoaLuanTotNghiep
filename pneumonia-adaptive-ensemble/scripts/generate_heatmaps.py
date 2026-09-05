"""Script to batch-generate XAI Grad-CAM and ViT Attention Heatmaps for test samples."""
import argparse
import os
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability.visualization import create_explanation_figure
from src.inference.predictor import PneumoniaPredictor
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Generate Explainable AI Heatmaps for Chest X-ray images.")
    parser.add_argument("--image_dir", type=str, default="data/raw", help="Directory of images to process")
    parser.add_argument("--output_dir", type=str, default="results/heatmaps", help="Directory to save figures")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples to generate")
    args = parser.parse_args()

    logger = setup_logger(name="generate_heatmaps", log_filename="generate_heatmaps.log")
    os.makedirs(args.output_dir, exist_ok=True)

    predictor = PneumoniaPredictor(mode="adaptive_ensemble")
    img_paths = list(Path(args.image_dir).glob("**/*.[jJ][pP][gG]")) + list(
        Path(args.image_dir).glob("**/*.[pP][nN][gG]")
    )

    if not img_paths:
        logger.warning(f"No images found in {args.image_dir}. Generating demo sample...")
        # Create a synthetic image for test demonstration
        demo_img = Image.new("RGB", (380, 380), color=(128, 128, 128))
        res, xai, pil_img = predictor.predict(demo_img, generate_xai=True)
        out_path = Path(args.output_dir) / "demo_explanation.png"
        create_explanation_figure(
            pil_img,
            xai,
            predicted_label=res["prediction"],
            confidence=res["confidence"],
            weights={"CNN": res["branch_details"]["cnn_weight"] or 0.5, "ViT": res["branch_details"]["vit_weight"] or 0.5},
            save_path=out_path,
        )
        logger.info(f"Saved demo explanation to {out_path}")
        return

    logger.info(f"Found {len(img_paths)} images. Processing first {args.max_samples}...")
    for idx, p in enumerate(img_paths[: args.max_samples]):
        res, xai, pil_img = predictor.predict(p, generate_xai=True)
        out_path = Path(args.output_dir) / f"xai_{idx:03d}_{p.stem}.png"
        create_explanation_figure(
            pil_img,
            xai,
            predicted_label=res["prediction"],
            confidence=res["confidence"],
            weights={"CNN": res["branch_details"]["cnn_weight"] or 0.5, "ViT": res["branch_details"]["vit_weight"] or 0.5},
            save_path=out_path,
        )
        logger.info(f"Saved explanation for [{p.name}] -> {out_path}")


if __name__ == "__main__":
    main()
