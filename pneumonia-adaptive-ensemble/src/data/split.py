import os
import re
import random
from collections import defaultdict

def extract_patient_id(filename):
    """
    Trích xuất mã định danh bệnh nhân (Patient ID) từ tên tệp ảnh Kaggle Chest X-Ray.
    Tránh hiện tượng rò rỉ dữ liệu (Data Leakage) giữa tập huấn luyện và tập kiểm định.
    
    Ví dụ:
    - person19_bacteria_50.jpeg -> person19
    - person100_virus_184.jpeg -> person100
    - NORMAL2-IM-0315-0001.jpeg -> NORMAL2-IM-0315
    - IM-0001-0001.jpeg -> IM-0001
    """
    base = os.path.splitext(filename)[0]
    
    # Dạng pneumonia: personXXXX_...
    match_person = re.match(r'^(person\d+)', base, re.IGNORECASE)
    if match_person:
        return match_person.group(1).lower()
        
    # Dạng normal: NORMAL2-IM-XXXX hoặc IM-XXXX
    match_normal2 = re.match(r'^(normal2-im-\d+)', base, re.IGNORECASE)
    if match_normal2:
        return match_normal2.group(1).lower()
        
    match_im = re.match(r'^(im-\d+)', base, re.IGNORECASE)
    if match_im:
        return match_im.group(1).lower()
        
    # Mặc định lấy phần trước dấu gạch dưới hoặc gạch ngang đầu tiên
    parts = re.split(r'[_\-]', base)
    if len(parts) > 1:
        return "-".join(parts[:2]).lower()
    return base.lower()

def create_patient_aware_split(root_dir, val_ratio=0.15, seed=42):
    """
    Tách 10-15% tập train làm tập validation theo Patient ID (bảo đảm không có 
    bệnh nhân nào vừa xuất hiện ở train vừa xuất hiện ở val).
    """
    random.seed(seed)
    train_dir = os.path.join(root_dir, 'train')
    
    split_info = {'train': [], 'val': []}
    class_map = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    for class_name, label in class_map.items():
        class_folder = os.path.join(train_dir, class_name)
        if not os.path.exists(class_folder):
            continue
            
        patient_images = defaultdict(list)
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                pid = extract_patient_id(fname)
                patient_images[pid].append(os.path.join(class_folder, fname))
                
        patients = list(patient_images.keys())
        random.shuffle(patients)
        
        n_val_patients = max(1, int(len(patients) * val_ratio))
        val_patients = set(patients[:n_val_patients])
        train_patients = set(patients[n_val_patients:])
        
        for pid in train_patients:
            for fpath in patient_images[pid]:
                split_info['train'].append((fpath, label, pid))
                
        for pid in val_patients:
            for fpath in patient_images[pid]:
                split_info['val'].append((fpath, label, pid))
                
    return split_info
