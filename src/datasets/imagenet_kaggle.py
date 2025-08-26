from typing import Optional, Dict, List, Tuple
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import kagglehub

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


class KaggleImageFolderImagenet(Dataset):
    """ImageNet-style folder dataset downloaded via kagglehub.

    Splits are expected to be 'train' or 'val'. Class subfolders will be used if present.
    """

    def __init__(self,
                 split: str = 'train',
                 resolution: int = 64,
                 kaggle_dataset_id: str = "ayaroshevskiy/downsampled-imagenet-64x64",
                 max_samples: Optional[int] = None,
                 random_subset_seed: Optional[int] = None,
                 random_flip_prob: float = 0.5):
        super().__init__()
        if split not in {"train", "val", "validation"}:
            raise ValueError("split must be 'train' or 'val'.")
        if split == "validation":
            split = "val"
        self.split = split
        self.resolution = resolution
        self.kaggle_dataset_id = kaggle_dataset_id
        self.max_samples = max_samples
        self.random_subset_seed = random_subset_seed
        self.random_flip_prob = float(random_flip_prob)

        self.samples: List[Tuple[pathlib.Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._download_and_scan()
        if not self.samples:
            raise RuntimeError(f"No images found for split '{self.split}' in Kaggle dataset '{self.kaggle_dataset_id}'.")
        if self.max_samples is not None:
            if self.random_subset_seed is not None:
                import random
                random.Random(self.random_subset_seed).shuffle(self.samples)
            self.samples = self.samples[: self.max_samples]

    def _download_and_scan(self):
        download_root = pathlib.Path(kagglehub.dataset_download(self.kaggle_dataset_id))
        candidate_split_dirs: List[pathlib.Path] = []
        candidate_split_dirs.append(download_root / self.split)
        dataset_slug = self.kaggle_dataset_id.split('/')[-1]
        candidate_split_dirs.append(download_root / dataset_slug / self.split)
        if self.split == 'val':
            candidate_split_dirs.append(download_root / 'validation')
            candidate_split_dirs.append(download_root / dataset_slug / 'validation')

        split_dir = None
        for cand in candidate_split_dirs:
            if cand.is_dir() and any(p.is_dir() for p in cand.iterdir()):
                split_dir = cand
                break
        if split_dir is None:
            possible_dirs = [d for d in download_root.rglob('*') if d.is_dir() and self.split in d.name.lower()]
            for cand in possible_dirs:
                has_images = any(p.suffix.lower() in _IMAGE_EXTS for p in cand.rglob('*'))
                if has_images:
                    split_dir = cand
                    break
        if split_dir is None:
            raise FileNotFoundError(f"Could not locate '{self.split}' split under {download_root}")

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
        if self.split == 'train' and (np.random.rand() < self.random_flip_prob):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_np = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(target_class_idx, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


# Backwards-compatible alias
KaggleImageFolderImagenet64 = KaggleImageFolderImagenet


