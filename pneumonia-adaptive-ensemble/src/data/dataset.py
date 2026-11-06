import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.data.split import create_patient_aware_split

class PneumoniaDataset(Dataset):
    """
    Dataset loader cho dữ liệu Chest X-Ray.
    Đã khắc phục 2 lỗi theo nhận xét phản biện:
    1. Bỏ RandomHorizontalFlip (tránh đảo ngược vị trí bóng tim từ trái sang phải).
    2. Khắc phục vấn đề tập val gốc chỉ có 16 ảnh: Tự động tách 15% tập train theo Patient ID
       để tránh rò rỉ dữ liệu bệnh nhân (Patient Leakage).
    """
    def __init__(self, root_dir, split='train', transform=None, img_size=224, val_ratio=0.15, seed=42):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.image_paths = []
        self.labels = []
        self.patient_ids = []
        
        class_map = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        if split == 'test':
            test_dir = os.path.join(root_dir, 'test')
            for class_name, label in class_map.items():
                class_dir = os.path.join(test_dir, class_name)
                if os.path.exists(class_dir):
                    for fname in os.listdir(class_dir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(class_dir, fname))
                            self.labels.append(label)
                            self.patient_ids.append(fname)
        else:
            # Kiểm tra xem tập val gốc của dataset có bị giới hạn (<= 16 ảnh) hay không
            val_dir = os.path.join(root_dir, 'val')
            n_raw_val = 0
            if os.path.exists(val_dir):
                for c in class_map.keys():
                    cdir = os.path.join(val_dir, c)
                    if os.path.exists(cdir):
                        n_raw_val += len([f for f in os.listdir(cdir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Nếu tập val gốc chỉ có 16 ảnh (hoặc không đủ lớn), chia lại theo Patient ID từ tập train
            if n_raw_val <= 16:
                splits = create_patient_aware_split(root_dir, val_ratio=val_ratio, seed=seed)
                selected = splits.get(split, [])
                for fpath, label, pid in selected:
                    self.image_paths.append(fpath)
                    self.labels.append(label)
                    self.patient_ids.append(pid)
            else:
                target_dir = os.path.join(root_dir, split)
                for class_name, label in class_map.items():
                    class_dir = os.path.join(target_dir, class_name)
                    if os.path.exists(class_dir):
                        for fname in os.listdir(class_dir):
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.image_paths.append(os.path.join(class_dir, fname))
                                self.labels.append(label)
                                self.patient_ids.append(fname)
                                
        self.transform = transform or self.get_default_transform(split)
        
    def get_default_transform(self, split):
        # KHÔNG SỬ DỤNG transforms.RandomHorizontalFlip()
        # vì ảnh X-quang ngực có tính bất đối xứng giải phẫu học (tim nằm bên trái).
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomRotation(degrees=7),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
    def get_class_weights(self):
        """
        Tính trọng số nghịch đảo tần suất lớp để khắc phục mất cân bằng dữ liệu
        giúp cải thiện Specificity (Độ đặc hiệu) khi huấn luyện.
        """
        if not self.labels:
            return torch.tensor([1.0, 1.0])
        labels_tensor = torch.tensor(self.labels)
        n_total = len(labels_tensor)
        n_normal = (labels_tensor == 0).sum().item()
        n_pneu = (labels_tensor == 1).sum().item()
        
        w_normal = n_total / (2.0 * max(1, n_normal))
        w_pneu = n_total / (2.0 * max(1, n_pneu))
        return torch.tensor([w_normal, w_pneu], dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
