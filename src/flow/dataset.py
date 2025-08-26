import sys
import platform
import os
import pathlib # For robust path normalization
from typing import Optional, Dict, List, Tuple, Any
import random

if platform.system() == "Windows":
    class MockResource:
        """A mock resource module to prevent errors on Windows, where the `resource` module is not available."""
        RLIMIT_NOFILE = 7 # RLIMIT_NOFILE is usually an int, value doesn't strictly matter for this mock
        
        def getrlimit(self, resource_type):
            if resource_type == self.RLIMIT_NOFILE:
                # Return some plausible default values for open file limits (soft, hard)
                return (1024, 4096) 
            raise ValueError(f"MockResource: Unsupported resource type: {resource_type}")

        def setrlimit(self, resource_type, limits):
            # This mock won't actually change system limits on Windows
            # It just needs to exist to prevent an AttributeError
            print(f"MockResource: setrlimit for {resource_type} called with {limits}, but not implemented on Windows.")
            pass

    sys.modules['resource'] = MockResource()

import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow_datasets as tfds
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image # For loading images from files
import kagglehub # For the new download method

# Suppress TFDS info messages if desired
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TFDSImagenet(Dataset):
    """A PyTorch Dataset for TFDS downsampled_imagenet datasets."""

    def __init__(self,
                 split: str = 'train',
                 resolution: int = 64,
                 max_samples: int = None,
                 data_dir: str = None,
                 manual_tar_dir: str = None):
        """Initializes the dataset, downloading and preparing it if necessary."""
        super().__init__()
        self.resolution = resolution
        if resolution not in {32, 64}:
            raise ValueError("Resolution must be 32 or 64 for TFDSImagenet.")
        dataset_name = f'downsampled_imagenet/{resolution}x{resolution}'

        if manual_tar_dir is not None:
            if platform.system() == "Windows":
                manual_tar_dir = str(pathlib.Path(manual_tar_dir).resolve())
            tfds.download.manual_dir = manual_tar_dir

        builder = tfds.builder(dataset_name, data_dir=data_dir)
        try:
            builder.download_and_prepare()
        except Exception as e:
            print(f"TFDS download_and_prepare for {dataset_name} failed: {e}")
            print(f"Ensure manual_tar_dir='{manual_tar_dir}' is correct if TFDS cannot download.")
            raise

        ds = builder.as_dataset(split=split, shuffle_files=False)
        if max_samples is not None:
            ds = ds.take(max_samples)
        
        self.tfds = ds
        self.max_samples = max_samples

        # Count length
        if max_samples is not None:
            self._length = int(max_samples)
        else:
            base_split_name = split.split('[')[0]
            if base_split_name not in builder.info.splits:
                raise ValueError(f"Split '{base_split_name}' not found in builder. Available: {list(builder.info.splits.keys())}")
            
            info = builder.info.splits[base_split_name].num_examples
            
            # Handle slices like 'train[:10%]'
            if '[' in split and ']' in split:
                slice_str = split[split.find('[')+1:split.find(']')]
                # Simplified logic for percentage slices
                if slice_str.endswith('%'):
                    self._length = int(info * int(slice_str.strip()[:-1]) / 100.0)
                else: # e.g. 'train[:1000]'
                     self._length = int(slice_str.split(':')[-1]) if ':' in slice_str else int(slice_str)
            else:
                 self._length = int(info)

            if max_samples is not None:
                 self._length = min(self._length, int(max_samples))

        self._prepare_list()

    def _prepare_list(self):
        """Materializes the tf.data.Dataset into a Python list for indexed access."""
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
        """Retrieves a sample from the dataset at the given index."""
        img_np = self._images[idx]
        lbl = self._labels[idx]

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(lbl, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


# For backwards compatibility with the old train.py if it's ever used.
class TFDSImagenet64(TFDSImagenet):
    def __init__(self, **kwargs):
        super().__init__(resolution=64, **kwargs)

class TFDSImagenet32(TFDSImagenet):
    def __init__(self, **kwargs):
        super().__init__(resolution=32, **kwargs)


class KaggleImageFolderImagenet(Dataset):
    """A PyTorch Dataset that downloads and loads an ImageFolder-style dataset from Kaggle."""

    def __init__(self,
                 split: str = 'train',  # 'train' or 'val'
                 resolution: int = 64,
                 kaggle_dataset_id: str = "ayaroshevskiy/downsampled-imagenet-64x64",
                 max_samples: Optional[int] = None,
                 random_subset_seed: Optional[int] = None):
        """Initializes the dataset, downloading and scanning it from Kaggle."""
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

        # Will populate:
        self.samples: List[Tuple[pathlib.Path, int]] = []  # (img_path, class_idx)
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._download_and_scan()

        if not self.samples:
            raise RuntimeError(f"No images found for split '{self.split}' in Kaggle dataset '{self.kaggle_dataset_id}'.")

        if self.max_samples is not None:
            # If a seed is provided, shuffle the dataset for a consistent random subset
            if self.random_subset_seed is not None:
                print(f"Shuffling dataset with seed {self.random_subset_seed} before taking a subset of {self.max_samples} samples.")
                random.Random(self.random_subset_seed).shuffle(self.samples)
            
            self.samples = self.samples[: self.max_samples]

    def _download_and_scan(self):
        """Downloads the dataset from Kaggle and scans the directory to find image samples."""
        print(f"Attempting to download '{self.kaggle_dataset_id}' using kagglehub...")
        download_root = pathlib.Path(kagglehub.dataset_download(self.kaggle_dataset_id))
        print(f"Kagglehub download completed. Content cached at: {download_root}")

        # Potential locations of the split directory
        candidate_split_dirs: List[pathlib.Path] = []

        # 1. Directly under download_root / split
        candidate_split_dirs.append(download_root / self.split)

        # 2. Download root / dataset_slug / split
        dataset_slug = self.kaggle_dataset_id.split('/')[-1]
        candidate_split_dirs.append(download_root / dataset_slug / self.split)

        # 3. Sometimes 'validation' is used instead of 'val'
        if self.split == 'val':
            candidate_split_dirs.append(download_root / 'validation')
            candidate_split_dirs.append(download_root / dataset_slug / 'validation')

        # Select the first directory that exists and contains subdirectories (classes)
        split_dir = None
        for cand in candidate_split_dirs:
            if cand.is_dir() and any(p.is_dir() for p in cand.iterdir()):
                split_dir = cand
                break
        if split_dir is None:
            # Fallback: recursively search for a directory whose name contains the split string (e.g. 'train')
            possible_dirs = [d for d in download_root.rglob('*') if d.is_dir() and self.split in d.name.lower()]
            for cand in possible_dirs:
                # Check if this directory directly contains images OR contains subdirs with images
                has_images = any(p.suffix.lower() in _IMAGE_EXTS for p in cand.rglob('*'))
                if has_images:
                    split_dir = cand
                    break

        if split_dir is None:
            raise FileNotFoundError(
                f"Could not locate '{self.split}' data in downloaded dataset structure under {download_root}. Searched recursively but found no directory with images.")

        print(f"Using split directory: {split_dir}")

        # Determine if class subdirectories exist
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if subdirs and all(any((p.suffix.lower() in _IMAGE_EXTS) for p in sd.rglob('*')) for sd in subdirs):
            # Treat each subdir as a class
            self.classes = sorted([d.name for d in subdirs])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            for cls_name in self.classes:
                cls_dir = split_dir / cls_name
                cls_idx = self.class_to_idx[cls_name]
                for img_path in cls_dir.rglob('*'):
                    if img_path.suffix.lower() in _IMAGE_EXTS:
                        self.samples.append((img_path, cls_idx))
        else:
            # No class folders – treat all images in split_dir (recursively) as a single class 0
            self.classes = ['unknown']
            self.class_to_idx = {'unknown': 0}
            for img_path in split_dir.rglob('*'):
                if img_path.suffix.lower() in _IMAGE_EXTS:
                    self.samples.append((img_path, 0))

        print(f"Found {len(self.samples)} images across {len(self.classes)} class folders for split '{self.split}'.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieves an image and its corresponding label from the dataset."""
        img_path, target_class_idx = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning a placeholder.")
            img_tensor = torch.zeros((3, self.resolution, self.resolution), dtype=torch.uint8)
            label_tensor = torch.tensor(-1, dtype=torch.long)
            return {"image": img_tensor, "label": label_tensor}

        # Geometric preprocessing: resize shorter side -> resolution (keep aspect), then center-crop to square
        from src.utils.image import aspect_preserving_resize_and_center_crop
        img = aspect_preserving_resize_and_center_crop(img, self.resolution)
        img_np = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(target_class_idx, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


# Allowed image extensions
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}

# For backwards compatibility
KaggleImageFolderImagenet64 = KaggleImageFolderImagenet


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
        # img is PIL.Image in RGB 32x32
        img_np = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


class ImageNet21kFolder(Dataset):
    """Generic ImageNet-21k style folder loader with class subfolders.

    Expects a directory structure like:
      root_dir/
        train/
          class_a/ *.jpg
          class_b/ *.jpg
        val/   (optional)
          class_a/ *.jpg
          ...
    """
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
            # Flat directory: all images belong to class 0
            self.classes = ['unknown']
            self.class_to_idx = {'unknown': 0}
            for img_path in split_dir.rglob('*'):
                if img_path.suffix.lower() in _IMAGE_EXTS:
                    self.samples.append((img_path, 0))

        if not self.samples:
            raise RuntimeError(f"No images found under {split_dir}")

        if self.max_samples is not None:
            if self.random_subset_seed is not None:
                print(f"Shuffling ImageNet21kFolder with seed {self.random_subset_seed} before subsetting {self.max_samples} samples.")
                random.Random(self.random_subset_seed).shuffle(self.samples)
            self.samples = self.samples[: self.max_samples]

        print(f"ImageNet21kFolder: {len(self.samples)} images across {len(self.classes)} classes in split '{self.split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, target_class_idx = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning a placeholder.")
            img_tensor = torch.zeros((3, self.resolution, self.resolution), dtype=torch.uint8)
            label_tensor = torch.tensor(-1, dtype=torch.long)
            return {"image": img_tensor, "label": label_tensor}

        # Aspect-preserving resize (shorter side -> resolution) + center-crop
        from src.utils.image import aspect_preserving_resize_and_center_crop
        img = aspect_preserving_resize_and_center_crop(img, self.resolution)
        img_np = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(target_class_idx, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


if __name__ == '__main__':
    # --- Test TFDSImagenet64 (original test) ---
    print("Testing TFDSImagenet64 Dataset...")
    # Ensure test_manual_tar_dir is set appropriately for your environment
    test_manual_tar_dir_tfds = "./local_imagenet64_tars/"
    test_prepared_data_dir_tfds = None # Let TFDS use its default
    test_max_samples_tfds = 10

    # Normalize path for Windows robustness in TFDS
    if platform.system() == "Windows":
        test_manual_tar_dir_tfds = str(pathlib.Path(test_manual_tar_dir_tfds).resolve())

    if not os.path.exists(test_manual_tar_dir_tfds) or \
       not any(f.endswith('.tar') for f in os.listdir(test_manual_tar_dir_tfds)):
        print(f"WARNING: TFDS test_manual_tar_dir '{test_manual_tar_dir_tfds}' does not exist or has no .tar files. Skipping TFDS test.")
    else:
        try:
            print(f"Attempting to load TFDS with manual_tar_dir: {test_manual_tar_dir_tfds}")
            train_dataset_tfds = TFDSImagenet64(
                split='train[:1%]',
                data_dir=test_prepared_data_dir_tfds,
                manual_tar_dir=test_manual_tar_dir_tfds,
                max_samples=test_max_samples_tfds
            )
            print(f"TFDS: Number of training samples: {len(train_dataset_tfds)}")
            if len(train_dataset_tfds) > 0:
                sample_tfds = train_dataset_tfds[0]
                print(f"TFDS: Image sample shape: {sample_tfds['image'].shape}, dtype: {sample_tfds['image'].dtype}")
                print(f"TFDS: Label sample: {sample_tfds['label']}, dtype: {sample_tfds['label'].dtype}")
        except Exception as e:
            print(f"An error occurred during TFDS dataset testing: {e}")
            import traceback
            traceback.print_exc()

    # --- Test KaggleImageFolderImagenet64 ---
    print("\nTesting KaggleImageFolderImagenet64 Dataset...")
    # This test will attempt to download from Kaggle.
    # Ensure your Kaggle API credentials are set up if you haven't used kagglehub before.
    test_kaggle_dataset_id = "ayaroshevskiy/downsampled-imagenet-64x64"
    test_max_samples_npz = 10

    try:
        print(f"Attempting to load KaggleImageFolderImagenet64 with dataset_id: {test_kaggle_dataset_id}")
        
        dataset_npz_train = KaggleImageFolderImagenet64(
            split='train',
            kaggle_dataset_id=test_kaggle_dataset_id,
            max_samples=test_max_samples_npz
        )
        print(f"KaggleImageFolder: Number of training samples: {len(dataset_npz_train)}")

        if len(dataset_npz_train) > 0:
            sample_npz = dataset_npz_train[0]
            img_npz_sample = sample_npz['image']
            lbl_npz_sample = sample_npz['label']
            print(f"KaggleImageFolder: Image sample shape: {img_npz_sample.shape}, dtype: {img_npz_sample.dtype}")
            print(f"KaggleImageFolder: Label sample: {lbl_npz_sample}, dtype: {lbl_npz_sample.dtype}")
            # Check label range (should be 0-indexed)
            if lbl_npz_sample.item() < 0 or lbl_npz_sample.item() >= 1000:
                 print(f"WARNING: KaggleImageFolder label sample {lbl_npz_sample.item()} seems out of 0-999 range.")

            dataloader_npz = DataLoader(dataset_npz_train, batch_size=min(2, len(dataset_npz_train)), shuffle=True)
            batch_npz = next(iter(dataloader_npz))
            print(f"KaggleImageFolder: Batch images shape: {batch_npz['image'].shape}")
            print(f"KaggleImageFolder: Batch labels: {batch_npz['label']}")
        else:
            print("KaggleImageFolder dataset (train) is empty or failed to load.")

        # Test validation set as well
        dataset_npz_val = KaggleImageFolderImagenet64(
            split='val',
            kaggle_dataset_id=test_kaggle_dataset_id,
            max_samples=test_max_samples_npz
        )
        print(f"KaggleImageFolder: Number of validation samples: {len(dataset_npz_val)}")
        if len(dataset_npz_val) > 0:
            sample_val_npz = dataset_npz_val[0]
            print(f"KaggleImageFolder Val: Image shape: {sample_val_npz['image'].shape}, Label: {sample_val_npz['label']}")
        else:
            print("KaggleImageFolder dataset (val) is empty or failed to load.")

    except Exception as e:
        print(f"An error occurred during KaggleImageFolderImagenet64 testing: {e}")
        import traceback
        traceback.print_exc()
    # finally:
        # Kagglehub downloads to a central cache, so no specific cleanup needed here
        # unless you want to clear the kaggle cache manually.
        # print(f"KaggleImageFolder test finished.") 