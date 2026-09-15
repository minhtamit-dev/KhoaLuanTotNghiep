import os
import sys
import shutil
import kagglehub

def is_valid_dataset(path):
    normal_dir = os.path.join(path, 'test', 'NORMAL')
    pneu_dir = os.path.join(path, 'test', 'PNEUMONIA')
    if os.path.exists(normal_dir) and os.path.exists(pneu_dir):
        if len(os.listdir(normal_dir)) > 0 and len(os.listdir(pneu_dir)) > 0:
            return True
    return False

def prepare_dataset():
    print("[+] Checking / Downloading Kaggle Chest X-Ray Pneumonia Dataset...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'data', 'chest_xray')
    
    if os.path.exists(dataset_dir) and is_valid_dataset(dataset_dir):
        print(f"[+] Dataset is validly loaded at: {dataset_dir}")
        return dataset_dir
        
    if os.path.exists(dataset_dir) or os.path.islink(dataset_dir):
        if os.path.islink(dataset_dir):
            os.unlink(dataset_dir)
        else:
            shutil.rmtree(dataset_dir)
            
    raw_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"[+] Downloaded raw dataset to: {raw_path}")
    
    target_data = None
    for root, dirs, files in os.walk(raw_path):
        if '__MACOSX' in root:
            continue
        if is_valid_dataset(root):
            target_data = root
            break
            
    if target_data is None:
        raise RuntimeError("Could not locate valid chest_xray dataset in downloaded files.")
        
    print(f"[+] Found valid dataset directory at: {target_data}")
    os.makedirs(os.path.dirname(dataset_dir), exist_ok=True)
    os.symlink(target_data, dataset_dir)
        
    print(f"[+] Dataset successfully linked to: {dataset_dir}")
    return dataset_dir

if __name__ == '__main__':
    prepare_dataset()
