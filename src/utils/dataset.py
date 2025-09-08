import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import platform
import os
import pathlib
import tensorflow_datasets as tfds
import torchvision
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List
from types import SimpleNamespace
import json
from pathlib import Path
import wandb
import time
import os
import requests
from io import BytesIO
import hashlib
from datasets import load_dataset
from huggingface_hub import login as hf_login
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
from src.utils.logging import get_logger
logger = get_logger(__name__)
from sentencepiece import SentencePieceProcessor
from src.utils.tokenizer import download_sentencepiece_model
import kagglehub

class LAIONPOPTextImageDataset(Dataset): # tokenizer: gs://t5-data/vocabs/cc_en.32000/sentencepiece.model
    def _download_tokenizer_model(self, tokenizer_path):
        """Download tokenizer model if it's a Google Storage URL"""
        if tokenizer_path.startswith('gs://'):
            # Convert gs:// URL to public HTTP URL and download
            cache_path = self.cache_dir / "sentencepiece.model"
            
            if not cache_path.exists():
                print(f"Downloading tokenizer model to {cache_path}...")
                
                # Convert gs:// URL to public HTTP URL
                public_url = tokenizer_path.replace('gs://', 'https://storage.googleapis.com/')
                
                try:
                    import urllib.request
                    print(f"Downloading from {public_url}")
                    urllib.request.urlretrieve(public_url, cache_path)
                    print(f"Successfully downloaded tokenizer model to {cache_path}")
                except Exception as e:
                    print(f"Failed to download tokenizer from {public_url}: {e}")
                    print("Please manually download the SentencePiece model and provide a local path")
                    raise e
            
            return str(cache_path)
        else:
            return tokenizer_path

    def __init__(
        self, 
        vocab_size=32000,
        tokenizer_path="gs://t5-data/vocabs/cc_en.32000/sentencepiece.model",
        max_text_len=64, 
        image_size=(256, 256),
        cache_dir="./laion_pop_cache",
        max_samples=10000,
        num_workers=4,
        timeout=10,
        use_cogvlm_captions=True,
        min_resolution=512,
        eos_id=1,
        bos_id=32000,
        ignore_pad=False,
    ):
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_samples = max_samples
        self.num_workers = num_workers
        self.timeout = timeout
        self.use_cogvlm_captions = use_cogvlm_captions
        self.min_resolution = min_resolution
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.ignore_pad = ignore_pad

        # Download/prepare tokenizer model via centralized util
        # Centralize tokenizer model download via src.tokenizer
        local_tokenizer_path = tokenizer_path
        if tokenizer_path.startswith('gs://'):
            try:
                local_tokenizer_path = download_sentencepiece_model()
            except Exception:
                # Fallback to class-local downloader (kept temporarily for robustness)
                local_tokenizer_path = self._download_tokenizer_model(tokenizer_path)
        
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.Load(local_tokenizer_path)
        if self.tokenizer.vocab_size() != self.vocab_size:
            raise ValueError(f"Vocab size mismatch: {self.tokenizer.vocab_size()} != {self.vocab_size}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        
        logger.info("Loading LAION-POP dataset...")
        # Attempt auto-auth from env/token file for gated datasets
        try:
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
            if hf_token:
                hf_login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass
        self.data = self._load_laion_pop_dataset()
        print(f"Loaded {len(self.data)} samples from LAION-POP dataset")
        
        self._predownload_images()
    
    def _load_laion_pop_dataset(self):
        """Load LAION-POP dataset from Hugging Face"""
        try:
            logger.info("Downloading LAION-POP dataset from Hugging Face...")
            
            # Load the full dataset - LAION-POP is relatively small (600k samples)
            dataset = load_dataset("laion/laion-pop", split="train", streaming=False)
            
            logger.info(f"Raw dataset size: {len(dataset)}")
            
            # Filter and collect samples
            data = []
            count = 0
            filtered_count = 0
            
            logger.info("Filtering LAION-POP samples...")
            for item in dataset:
                try:
                    width = item.get('width', 0)
                    height = item.get('height', 0)
                    min_dim = min(width, height) if width and height else 0
                    
                    text = None
                    if self.use_cogvlm_captions and 'cogvlm_caption' in item:
                        text = item['cogvlm_caption']
                    elif 'llava_caption' in item:
                        text = item['llava_caption']
                    elif 'alt_text' in item:
                        text = item['alt_text']
                    elif 'TEXT' in item:
                        text = item['TEXT']
                    
                    if (text and len(text.strip()) > 10 and len(text.strip()) < 500 and
                        item.get('url') and item['url'].startswith('http') and
                        min_dim >= self.min_resolution):
                        
                        data.append({
                            'text': text.strip(),
                            'url': item['url'],
                            'width': width,
                            'height': height,
                            'key': item.get('key', str(count)),
                            'alt_text': item.get('alt_text', text),
                            'cogvlm_caption': item.get('cogvlm_caption', ''),
                            'llava_caption': item.get('llava_caption', ''),
                        })
                        count += 1
                        
                        if count >= self.max_samples:
                            break
                    else:
                        filtered_count += 1
                        
                    if (count + filtered_count) % 10000 == 0 and count + filtered_count > 0:
                        logger.info(f"Processed {count + filtered_count} samples, kept {count}, filtered {filtered_count}")
                        
                except Exception as e:
                    filtered_count += 1
                    continue
            
            logger.info(f"Filtering complete: {count} samples kept, {filtered_count} filtered out")
            return data
            
        except Exception as e:
            logger.error(f"Error loading LAION-POP dataset: {e}")
            raise e
    
    def _get_cache_path(self, url):
        """Generate cache path for image URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.jpg"
    
    def _download_image(self, url, cache_path, max_retries=2):
        """Download image with retries and error handling"""
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert('RGB')
            except:
                cache_path.unlink()
        
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
                response = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
                #print(response)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    continue
                
                image_data = BytesIO(response.content)
                image = Image.open(image_data).convert('RGB')
                
                if image.size[0] < 64 or image.size[1] < 64:
                    continue
                
                try:
                    image.save(cache_path, 'JPEG', quality=85)
                except Exception:
                    pass
                
                return image
                
            except Exception as e:
                if attempt == max_retries - 1:
                    pass
                time.sleep(0.5)
        
        return None
    
    def _predownload_images(self, num_predownload=200):
        """Pre-download some images for faster training startup"""
        logger.info(f"Pre-downloading {num_predownload} images...")
        
        def download_worker(idx):
            if idx >= len(self.data):
                return False
            
            item = self.data[idx]
            if item['url'] == 'dummy':
                return True
                
            cache_path = self._get_cache_path(item['url'])
            if cache_path.exists():
                return True
                
            image = self._download_image(item['url'], cache_path)
            # debug: logger.debug(image)
            return image is not None
        
        with ThreadPoolExecutor(max_workers=min(8, self.num_workers)) as executor:
            futures = [executor.submit(download_worker, i) for i in range(min(num_predownload, len(self.data)))]
            
            success_count = 0
            for i, future in enumerate(as_completed(futures)):
                if future.result():
                    success_count += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Pre-downloaded {i + 1}/{num_predownload} images ({success_count} successful)")
        
        logger.info(f"Pre-download completed: {success_count}/{num_predownload} successful")
    
    def __len__(self):
        return len(self.data)
    
    def tokenize_text(self, text):
        text = text.strip().lower()
        
        prefixes_to_remove = [
            "this image shows", "this image depicts", "the image shows",
            "the image depicts", "this is an image of", "this is a photo of",
            "in this image", "the photo shows"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                if text.startswith(','):
                    text = text[1:].strip()
                break
        
        # Tokenize using SentencePiece (without EOS)
        tokens = self.tokenizer.EncodeAsIds(text)
        
        # Add EOS token manually
        tokens = tokens + [self.eos_id]
        
    
        # True for real tokens, False for pads
        text_mask = [1] * len(tokens)
        text_loss = [1] * len(tokens)
        # Use 0 as pad id
        pad_value = 0
        
        # Truncate if too long
        if len(tokens) > self.max_text_len:
            tokens = tokens[:self.max_text_len]
            text_mask = text_mask[:self.max_text_len]
            text_loss = text_loss[:self.max_text_len]
        else:
            # Pad to max_text_len
            padding_len = self.max_text_len - len(tokens)
            tokens.extend([pad_value] * padding_len)
            # Pads are masked out
            text_mask.extend([0] * padding_len)
            text_loss.extend([0] * padding_len)
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'text_mask': torch.tensor(text_mask, dtype=torch.bool),
            'text_loss': torch.tensor(text_loss, dtype=torch.bool)
        }
    
    def __getitem__(self, idx):
        max_retries = 10  # Maximum number of different samples to try
        original_idx = idx
        
        for attempt in range(max_retries):
            try:
                # Use modulo to wrap around if we exceed dataset length
                current_idx = (original_idx + attempt) % len(self.data)
                item = self.data[current_idx]
                text = item['text']
                url = item['url']
                key = item.get('key', str(current_idx))
                
                tokenization_result = self.tokenize_text(text)
                
                image = None
                
                if url == 'dummy':
                    # Skip dummy URLs and try next sample
                    continue
                else:
                    cache_path = self._get_cache_path(url)
                    image = self._download_image(url, cache_path, max_retries=3)
                
                if image is None:
                    # Skip failed downloads and try next sample
                    continue
                
                try:
                    image = self.transform(image)
                    image = (image * 2) - 1  # normalize to [-1, 1]
                except Exception as e:
                    # Skip failed image transformations and try next sample
                    continue
                
                # If we reach here, everything succeeded
                return {
                    'text': tokenization_result['tokens'],
                    'image': image,
                    'text_mask': tokenization_result['text_mask'],
                    'text_loss': tokenization_result['text_loss'],
                    'raw_text': text,
                    'key': key,
                    'alt_text': item.get('alt_text', text),
                    'cogvlm_caption': item.get('cogvlm_caption', ''),
                    'llava_caption': item.get('llava_caption', ''),
                }
                
            except Exception as e:
                # Skip any other errors and try next sample
                continue
        
        # If all retries failed, raise an error
        raise RuntimeError(f"Failed to load any valid sample after {max_retries} attempts starting from index {original_idx}")

class TinyStoriesDataset(Dataset):
    def __init__(
        self,
        vocab_size=32000,
        tokenizer_path="gs://t5-data/vocabs/cc_en.32000/sentencepiece.model",
        max_text_len=64,
        cache_dir="./tinystories_cache",
        split="train",
        max_samples=None,
        eos_id=1,
        bos_id=32000,
        ignore_pad=False,
    ):
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_samples = max_samples
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.ignore_pad = ignore_pad

        # Download/prepare tokenizer model via centralized util
        local_tokenizer_path = tokenizer_path
        if tokenizer_path.startswith('gs://'):
            try:
                local_tokenizer_path = download_sentencepiece_model()
            except Exception:
                local_tokenizer_path = self._download_tokenizer_model(tokenizer_path)
        
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.Load(local_tokenizer_path)
        if self.tokenizer.vocab_size() != self.vocab_size:
            raise ValueError(f"Vocab size mismatch: {self.tokenizer.vocab_size()} != {self.vocab_size}")
        
        logger.info(f"Loading TinyStories dataset ({split} split)...")
        self.data = self._load_tinystories_dataset(split)
        print(f"Loaded {len(self.data)} samples from TinyStories dataset")

    def _download_tokenizer_model(self, tokenizer_path):
        """Download tokenizer model if it's a Google Storage URL"""
        if tokenizer_path.startswith('gs://'):
            cache_path = self.cache_dir / "sentencepiece.model"
            
            if not cache_path.exists():
                print(f"Downloading tokenizer model to {cache_path}...")
                
                public_url = tokenizer_path.replace('gs://', 'https://storage.googleapis.com/')
                
                try:
                    import urllib.request
                    print(f"Downloading from {public_url}")
                    urllib.request.urlretrieve(public_url, cache_path)
                    print(f"Successfully downloaded tokenizer model to {cache_path}")
                except Exception as e:
                    print(f"Failed to download tokenizer from {public_url}: {e}")
                    print("Please manually download the SentencePiece model and provide a local path")
                    raise e
            
            return str(cache_path)
        else:
            return tokenizer_path

    def _load_tinystories_dataset(self, split):
        """Load TinyStories dataset from Hugging Face"""
        try:
            logger.info("Downloading TinyStories dataset from Hugging Face...")
            
            # Load the dataset
            dataset = load_dataset("roneneldan/TinyStories", split=split)
            
            logger.info(f"Raw dataset size: {len(dataset)}")
            
            # Filter and collect samples
            data = []
            count = 0
            
            print("Processing TinyStories samples...")
            # Use quarter of the samples
            # max_samples = 2000 if split=="train" else 200
            
            for item in dataset:
                try:
                    text = item.get('text', '').strip()
                    
                    if text and len(text) > 10:
                        data.append({
                            'text': text,
                            'key': str(count),
                        })
                        count += 1
                        
                        if self.max_samples and count >= self.max_samples:
                            break
                        
                    if count % 10000 == 0 and count > 0:
                        logger.info(f"Processed {count} samples")
                        
                except Exception as e:
                    continue
            
            logger.info(f"Processing complete: {count} samples loaded")
            return data
            
        except Exception as e:
            logger.error(f"Error loading TinyStories dataset: {e}")
            raise e

    def tokenize_text(self, text):
        text = text.strip()

        prefixes_to_remove = [
            "this image shows", "this image depicts", "the image shows",
            "the image depicts", "this is an image of", "this is a photo of",
            "in this image", "the photo shows"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                if text.startswith(','):
                    text = text[1:].strip()
                break
        
        # Tokenize using SentencePiece
        tokens = self.tokenizer.EncodeAsIds(text)
        
        # Add EOS token manually
        tokens = tokens + [self.eos_id]
        
        # True for real tokens, False for pads
        text_mask = [1] * len(tokens)
        text_loss = [1] * len(tokens)
        pad_value = 0
        
        # Truncate if too long
        if len(tokens) > self.max_text_len:
            tokens = tokens[:self.max_text_len]
            text_mask = text_mask[:self.max_text_len]
            text_loss = text_loss[:self.max_text_len]
        else:
            # Pad to max_text_len
            padding_len = self.max_text_len - len(tokens)
            tokens.extend([pad_value] * padding_len)
            text_mask.extend([0] * padding_len)
            text_loss.extend([0] * padding_len)
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'text_mask': torch.tensor(text_mask, dtype=torch.bool),
            'text_loss': torch.tensor(text_loss, dtype=torch.bool)
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        key = item['key']
        
        tokenization_result = self.tokenize_text(text)
        
        return {
            'input_ids': tokenization_result['tokens'],
            'attention_mask': tokenization_result['text_mask'],
            'labels': tokenization_result['tokens'], 
            'raw_text': text,
            'key': key,
        }


# ---- Centralized image dataset implementations (moved from src/flow/dataset.py) ----

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
            if platform.system() == "Windows":
                manual_tar_dir = str(pathlib.Path(manual_tar_dir).resolve())
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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


class ImageNet21kFolder(Dataset):
    """Generic ImageNet-21k style folder loader with class subfolders."""

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
                import random
                random.Random(self.random_subset_seed).shuffle(self.samples)
            self.samples = self.samples[: self.max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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


def create_datasets_and_loaders(config: SimpleNamespace, accelerator) -> Tuple[Any, Any, DataLoader, DataLoader]:
    """Create dataset, val_dataset and corresponding data loaders based on config and accelerator.

    Returns (dataset, val_dataset, dataloader, val_loader).
    """
    # All dataset classes are defined in this module; avoid cross-module shim imports

    dataset_choice = getattr(config, 'dataset')
    if str(dataset_choice).lower() == 'imagenet64_kaggle':
        H, W = tuple(getattr(config, 'input_size'))
        res = int(H)
        dataset = KaggleImageFolderImagenet(
            split='train',
            resolution=res,
            kaggle_dataset_id=getattr(config, 'kaggle_dataset_id'),
            max_samples=getattr(config, 'max_samples'),
            random_flip_prob=float(getattr(config, 'random_flip_prob'))
        )
        val_dataset = KaggleImageFolderImagenet(
            split='val', resolution=res,
            kaggle_dataset_id=getattr(config, 'kaggle_dataset_id'),
            max_samples=getattr(config, 'max_samples', None),
            random_flip_prob=0.0
        )
    elif str(dataset_choice).lower() == 'imagenet21k_folder':
        root = getattr(config, 'imagenet21k_root', None)
        if not root:
            raise ValueError("--imagenet21k_root must be provided for imagenet21k_folder dataset")
        H, W = tuple(getattr(config, 'input_size'))
        res = int(H)
        dataset = ImageNet21kFolder(root_dir=root, split='train', resolution=res, max_samples=getattr(config, 'max_samples', None))
        val_dataset = ImageNet21kFolder(root_dir=root, split='val', resolution=res, max_samples=getattr(config, 'max_samples', None))
    else:
        dataset = LAIONPOPTextImageDataset(
            vocab_size=getattr(config, 'vocab_size'),
            tokenizer_path=getattr(config, 'tokenizer_path'),
            max_text_len=getattr(config, 'max_seq_len'),
            image_size=tuple(getattr(config, 'input_size')),
            cache_dir=getattr(config, 'cache_dir'),
            max_samples=getattr(config, 'max_samples'),
            use_cogvlm_captions=getattr(config, 'use_cogvlm_captions'),
            min_resolution=getattr(config, 'min_resolution'),
            num_workers=getattr(config, 'num_workers'),
            ignore_pad=getattr(config, 'ignore_pad')
        )
        # No explicit val set; reuse train dataset for a quick sanity val (not ideal)
        val_dataset = dataset

    train_sampler, val_sampler = accelerator.build_samplers(dataset, val_dataset)
    pin_mem = True if accelerator.device.type == 'cuda' else False

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(getattr(config, 'num_workers')),
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
        pin_memory=pin_mem
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(getattr(config, 'num_workers')),
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=False,
        pin_memory=pin_mem
    )

    return dataset, val_dataset, dataloader, val_loader