import os
import sys
import subprocess

def run_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)
    
    python_bin = sys.executable
    print(f"============================================================")
    print(f"🚀 RUNNING ALL EXPERIMENT STEPS AUTOMATICALLY WITH {python_bin}")
    print(f"============================================================")
    
    scripts = [
        ('1. Prepare Dataset', 'scripts/prepare_dataset.py'),
        ('2. Train Models (PyTorch GPU)', 'scripts/train_models.py'),
        ('3. Evaluate Models & Ensembles', 'scripts/evaluate_models.py'),
        ('4. Generate Grad-CAM Heatmaps', 'scripts/generate_heatmaps.py'),
        ('5. Generate PDF Report', 'scripts/generate_pdf_report.py')
    ]
    
    for title, script_rel in scripts:
        print(f"\n---> {title}...")
        script_path = os.path.join(base_dir, script_rel)
        cmd = [python_bin, script_path]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[!] Error executing {script_rel}. Exiting pipeline.")
            sys.exit(res.returncode)
            
    print(f"\n============================================================")
    print(f"✅ ALL EXPERIMENT STEPS COMPLETED SUCCESSFULLY!")
    print(f"📄 Report generated: CNTT-KLCN171_NguyenGiaKhang.pdf")
    print(f"============================================================")

if __name__ == '__main__':
    run_all()
