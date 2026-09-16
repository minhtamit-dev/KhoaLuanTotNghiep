import os
import sys
import subprocess

import argparse

def run_all():
    parser = argparse.ArgumentParser(description="Run all experiment steps")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)
    
    python_bin = sys.executable
    print(f"============================================================")
    print(f"🚀 RUNNING ALL EXPERIMENT STEPS AUTOMATICALLY WITH {python_bin}")
    print(f"============================================================")
    
    scripts = [
        ('1. Prepare Dataset', [python_bin, os.path.join(base_dir, 'scripts/prepare_dataset.py')]),
        ('2. Train Models (PyTorch GPU)', [python_bin, os.path.join(base_dir, 'scripts/train_models.py'), '--epochs', str(args.epochs)]),
        ('3. Evaluate Models & Ensembles', [python_bin, os.path.join(base_dir, 'scripts/evaluate_models.py')]),
        ('4. Generate Grad-CAM Heatmaps', [python_bin, os.path.join(base_dir, 'scripts/generate_heatmaps.py')]),
        ('5. Generate PDF Report', [python_bin, os.path.join(base_dir, 'scripts/generate_pdf_report.py')])
    ]
    
    for title, cmd in scripts:
        print(f"\n---> {title}...")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[!] Error executing {cmd}. Exiting pipeline.")
            sys.exit(res.returncode)
            
    print(f"\n============================================================")
    print(f"✅ ALL EXPERIMENT STEPS COMPLETED SUCCESSFULLY!")
    print(f"📄 Report generated: CNTT-KLCN171_NguyenGiaKhang.pdf")
    print(f"============================================================")

if __name__ == '__main__':
    run_all()
