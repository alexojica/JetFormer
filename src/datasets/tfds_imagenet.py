from typing import Optional

import torch
from torch.utils.data import Dataset
import tensorflow_datasets as tfds


class TFDSImagenet(Dataset):
    """A PyTorch Dataset for TFDS downsampled_imagenet datasets (32x32 or 64x64)."""

    def __init__(self,
                 split: str = 'train',
                 resolution: int = 64,
                 max_samples: Optional[int] = None,
                 data_dir: str = None,
                 manual_tar_dir: str = None):
        super().__init__()
        self.resolution = resolution
        if resolution not in {32, 64}:
            raise ValueError("Resolution must be 32 or 64 for TFDSImagenet.")
        dataset_name = f'downsampled_imagenet/{resolution}x{resolution}'

        if manual_tar_dir is not None:
            tfds.download.manual_dir = manual_tar_dir

        builder = tfds.builder(dataset_name, data_dir=data_dir)
        builder.download_and_prepare()
        ds = builder.as_dataset(split=split, shuffle_files=False)
        if max_samples is not None:
            ds = ds.take(max_samples)
        self.tfds = ds
        self._length = int(max_samples) if max_samples is not None else builder.info.splits[split.split('[')[0]].num_examples

        self._prepare_list()

    def _prepare_list(self):
        self._images = []
        self._labels = []
        count = 0
        for example in tfds.as_numpy(self.tfds):
            img_uint8 = example['image']
            lbl = example['label']
            self._images.append(img_uint8)
            self._labels.append(int(lbl))
            count += 1
            if self._length is not None and count >= self._length:
                break
        self._length = count

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        img_np = self._images[idx]
        lbl = self._labels[idx]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(lbl, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


class TFDSImagenet64(TFDSImagenet):
    def __init__(self, **kwargs):
        super().__init__(resolution=64, **kwargs)


class TFDSImagenet32(TFDSImagenet):
    def __init__(self, **kwargs):
        super().__init__(resolution=32, **kwargs)


