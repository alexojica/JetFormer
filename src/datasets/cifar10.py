import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np


class TorchvisionCIFAR10(Dataset):
    """Wrapper around torchvision CIFAR-10 to return uint8 CHW images."""
    def __init__(self, split: str = 'train', download: bool = True):
        super().__init__()
        train = (split == 'train')
        self.ds = torchvision.datasets.CIFAR10(
            root="./data/cifar10",
            train=train,
            download=download,
            transform=None,
            target_transform=None,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]
        img_np = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


