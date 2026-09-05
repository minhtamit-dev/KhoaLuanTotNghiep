"""Master orchestrator script running the entire research pipeline from A to Z."""
import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger


def run_command(cmd_list: list, logger, step_name: str):
    logger.info(f"\n==========================================")
    logger.info(f"▶ STARTING: {step_name}")
    logger.info(f"Command: {' '.join(cmd_list)}")
    logger.info(f"==========================================")
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        logger.error(f"❌ Step '{step_name}' failed with return code {res.returncode}")
        return False
    logger.info(f"✅ COMPLETED: {step_name}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Full Research Pipeline")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and jump directly to evaluation")
    args = parser.parse_args()

    logger = setup_logger(name="run_pipeline", log_filename="run_pipeline.log")
    python_exec = sys.executable

    # Step 1: Prepare dataset
    if not run_command([python_exec, "scripts/prepare_dataset.py"], logger, "1. Dataset Preparation & Splitting"):
        return

    # Step 2 & 3: Training
    if not args.skip_train:
        if not run_command([python_exec, "scripts/train_efficientnet.py"], logger, "2. Train EfficientNet-B4 Baseline"):
            return
        if not run_command([python_exec, "scripts/train_vit.py"], logger, "3. Train ViT-B/16 Baseline"):
            return

    # Step 4: Evaluation
    if not run_command([python_exec, "scripts/evaluate_models.py"], logger, "4. Comparative Multi-Model Evaluation"):
        return

    # Step 5: Explainability
    if not run_command([python_exec, "scripts/generate_heatmaps.py"], logger, "5. Generate Explainability Heatmaps"):
        return

    # Step 6: Export tables
    if not run_command([python_exec, "scripts/export_results.py"], logger, "6. Export Tables (CSV, MD, LaTeX)"):
        return

    logger.info("🎉 All research pipeline steps completed successfully!")


if __name__ == "__main__":
    main()
