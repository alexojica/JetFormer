from typing import Optional, Dict, List, Tuple
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


class ImageNet21kFolder(Dataset):
    """ImageNet-21k style folder loader with class subfolders."""

    def __init__(self, root_dir: str, split: str = 'train', resolution: int = 64, max_samples: Optional[int] = None, random_subset_seed: Optional[int] = None):
        super().__init__()
        self.root_dir = pathlib.Path(root_dir)
        if split not in {"train", "val", "validation"}:
            raise ValueError("split must be 'train' or 'val'.")
        if split == "validation":
            split = "val"
        self.split = split
        self.resolution = resolution
        self.max_samples = max_samples
        self.random_subset_seed = random_subset_seed

        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples: List[Tuple[pathlib.Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if subdirs and all(any((p.suffix.lower() in _IMAGE_EXTS) for p in sd.rglob('*')) for sd in subdirs):
            self.classes = sorted([d.name for d in subdirs])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            for cls_name in self.classes:
                cls_dir = split_dir / cls_name
                cls_idx = self.class_to_idx[cls_name]
                for img_path in cls_dir.rglob('*'):
                    if img_path.suffix.lower() in _IMAGE_EXTS:
                        self.samples.append((img_path, cls_idx))
        else:
            self.classes = ['unknown']
            self.class_to_idx = {'unknown': 0}
            for img_path in split_dir.rglob('*'):
                if img_path.suffix.lower() in _IMAGE_EXTS:
                    self.samples.append((img_path, 0))

        if not self.samples:
            raise RuntimeError(f"No images found under {split_dir}")

        if self.max_samples is not None:
            if self.random_subset_seed is not None:
                import random
                random.Random(self.random_subset_seed).shuffle(self.samples)
            self.samples = self.samples[: self.max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        img_path, target_class_idx = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img_tensor = torch.zeros((3, self.resolution, self.resolution), dtype=torch.uint8)
            label_tensor = torch.tensor(-1, dtype=torch.long)
            return {"image": img_tensor, "label": label_tensor}

        from src.utils.image import aspect_preserving_resize_and_center_crop
        img = aspect_preserving_resize_and_center_crop(img, self.resolution)
        img_np = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(target_class_idx, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


