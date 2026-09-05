"""Script to aggregate metrics and export comparative tables to CSV and LaTeX."""
import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger


def main():
    logger = setup_logger(name="export_results", log_filename="export_results.log")
    metrics_dir = Path("results/metrics")
    tables_dir = Path("results/tables")
    os.makedirs(tables_dir, exist_ok=True)

    json_files = list(metrics_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No metric JSON files found in {metrics_dir}.")
        return

    rows = []
    for f in json_files:
        model_name = f.stem
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        rows.append({
            "Model": model_name,
            "Accuracy": f"{data.get('accuracy', 0.0):.4f}",
            "Precision": f"{data.get('precision', 0.0):.4f}",
            "Recall (Sensitivity)": f"{data.get('recall', 0.0):.4f}",
            "Specificity": f"{data.get('specificity', 0.0):.4f}",
            "F1-Score": f"{data.get('f1_score', 0.0):.4f}",
            "ROC-AUC": f"{data.get('roc_auc', 0.0):.4f}",
            "ECE": f"{data.get('ece', 0.0):.4f}",
        })

    df = pd.DataFrame(rows)
    # Sort by F1-Score descending
    df = df.sort_values(by="F1-Score", ascending=False)

    # 1. Export CSV
    csv_path = tables_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV comparison table to {csv_path}")

    # 2. Export Markdown table
    md_path = tables_dir / "model_comparison.md"
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("# Model Performance Comparison\n\n")
        fp.write(df.to_markdown(index=False))
        fp.write("\n")
    logger.info(f"Saved Markdown table to {md_path}")

    # 3. Export LaTeX table for Thesis report
    latex_path = tables_dir / "model_comparison.tex"
    latex_code = df.to_latex(index=False, caption="So sánh hiệu năng giữa các mô hình và Ensemble", label="tab:model_comparison")
    with open(latex_path, "w", encoding="utf-8") as fp:
        fp.write(latex_code)
    logger.info(f"Saved LaTeX table for Thesis to {latex_path}")


if __name__ == "__main__":
    main()
