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
 # Defer importing TFDS until we have disabled TF GPU usage (see helper below)
import torchvision
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List, Union
from types import SimpleNamespace
import json
from pathlib import Path
import wandb
import time
import os
import requests
from io import BytesIO
import hashlib
from datasets import load_dataset, Image as HFImage
from huggingface_hub import login as hf_login
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
from src.utils.logging import get_logger
logger = get_logger(__name__)
from sentencepiece import SentencePieceProcessor
from src.utils.tokenizer import download_sentencepiece_model
import random

def _import_tfds_cpu_only():
    """Import tensorflow_datasets with TensorFlow forced to use CPU only and not pre-allocate GPU VRAM.

    Returns the imported tensorflow_datasets module.
    """
    try:
        # Minimize TF logging/noise and disable onednn to avoid CPU kernel warnings
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
        # Allow growth in case TF ends up using GPU (shouldn't after we hide it)
        os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
        # Import TF and hide GPUs before TF runtime is initialized
        import tensorflow as tf  # type: ignore
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
    except Exception:
        # If TF is not available or fails to import, tfds import may still work for some ops
        pass
    import tensorflow_datasets as tfds  # type: ignore
    return tfds

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

        tfds = _import_tfds_cpu_only()
        self._tfds = tfds
        builder = self._tfds.builder(dataset_name, data_dir=data_dir)
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
        for example in self._tfds.as_numpy(self.tfds):
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


class TFDSImagenetResized(Dataset):
    """TFDS imagenet_resized/64x64 wrapper returning {image:uint8 CHW, label}."""

    def __init__(self,
                 split: str = 'train',
                 resolution: int = 64,
                 max_samples: Optional[int] = None,
                 data_dir: str = None,
                 class_subset: Optional[Union[int, str]] = None):
        super().__init__()
        if resolution != 64:
            raise ValueError("TFDSImagenetResized currently supports only 64x64 resolution.")
        if split == 'val':
            split = 'validation'
        tfds = _import_tfds_cpu_only()
        self._tfds = tfds
        self._split = split
        self._builder = self._tfds.builder('imagenet_resized/64x64', data_dir=data_dir)
        self._builder.download_and_prepare()
        # Only pre-truncate when no class subset is requested
        if class_subset is None and max_samples is not None:
            self.tfds = self._builder.as_dataset(split=split, shuffle_files=False).take(max_samples)
        else:
            self.tfds = self._builder.as_dataset(split=split, shuffle_files=False)
        self._length = int(max_samples) if max_samples is not None else self._builder.info.splits[split].num_examples
        self._max_samples = int(max_samples) if max_samples is not None else None
        self.class_subset = class_subset
        self._prepare_list()

    def _prepare_list(self):
        self._images: List[np.ndarray] = []
        self._labels: List[int] = []
        # Provide class-like attributes for compatibility with helpers
        self.classes = list(range(1000))
        count = 0
        # Default path: populate from (possibly pre-truncated) dataset
        if self.class_subset is None or self._max_samples is None:
            for example in self._tfds.as_numpy(self.tfds):
                self._images.append(example['image'])
                self._labels.append(int(example['label']))
                count += 1
                if self._length is not None and count >= self._length:
                    break
            self._length = count
            self.samples = [(i, self._labels[i]) for i in range(self._length)]
            return

        # Resolve desired class id
        cid: Optional[int] = None
        try:
            if isinstance(self.class_subset, str):
                if self.class_subset.isdigit():
                    cid = int(self.class_subset)
            elif isinstance(self.class_subset, int):
                cid = int(self.class_subset)
        except Exception:
            cid = None

        if cid is None or not (0 <= cid < len(self.classes)):
            # Fallback to default behavior
            for example in self._tfds.as_numpy(self.tfds):
                self._images.append(example['image'])
                self._labels.append(int(example['label']))
                count += 1
                if self._length is not None and count >= self._length:
                    break
            self._length = count
            self.samples = [(i, self._labels[i]) for i in range(self._length)]
            return

        # First pass: count how many samples belong to the target class (avoid image decode)
        try:
            meta_ds = self._builder.as_dataset(
                split=self._split,
                shuffle_files=False,
                decoders={"image": self._tfds.decode.SkipDecoding()},
            )
            total_in_class = 0
            for ex in self._tfds.as_numpy(meta_ds):
                try:
                    if int(ex['label']) == cid:
                        total_in_class += 1
                except Exception:
                    continue
        except Exception:
            total_in_class = None

        # If class has fewer than max_samples, collect only that class
        if total_in_class is not None and total_in_class < int(self._max_samples):
            full_ds = self._builder.as_dataset(split=self._split, shuffle_files=False)
            for ex in self._tfds.as_numpy(full_ds):
                try:
                    if int(ex['label']) == cid:
                        self._images.append(ex['image'])
                        self._labels.append(cid)
                except Exception:
                    continue
            self._length = len(self._labels)
            self.samples = [(i, self._labels[i]) for i in range(self._length)]
            return

        # Otherwise, keep default multi-class behavior capped by max_samples
        default_ds = self._builder.as_dataset(split=self._split, shuffle_files=False).take(int(self._max_samples))
        for example in self._tfds.as_numpy(default_ds):
            self._images.append(example['image'])
            self._labels.append(int(example['label']))
            count += 1
            if self._length is not None and count >= self._length:
                break
        self._length = count
        self.samples = [(i, self._labels[i]) for i in range(self._length)]

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_np = self._images[idx]
        lbl = self._labels[idx]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(lbl, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


class TFDSImagenetResized64(TFDSImagenetResized):
    def __init__(self, **kwargs):
        super().__init__(resolution=64, **kwargs)


class HFImagenet1k(Dataset):
    """Hugging Face ILSVRC/imagenet-1k wrapper returning {image:uint8 CHW, label}.

    Requires accepting dataset terms on Hugging Face and (optionally) providing
    an auth token via the HF_TOKEN/HUGGINGFACE_TOKEN environment variables.
    """
    def __init__(self,
                 split: str = 'train',
                 resolution: int = 256,
                 max_samples: Optional[int] = None,
                 class_subset: Optional[Union[int, str]] = None,
                 random_flip_prob: float = 0.0,
                 hf_cache_dir: Optional[str] = None,
                 safe_decode: bool = True):
        super().__init__()
        if split == 'val':
            split = 'validation'
        self.split = split
        self.resolution = int(resolution)
        self._max_samples = int(max_samples) if max_samples is not None else None
        self.class_subset = class_subset
        self._flip_prob = float(random_flip_prob) if split == 'train' else 0.0

        # Attempt auto-auth for gated dataset access
        try:
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
            if hf_token:
                hf_login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

        # Prepare custom cache directory if provided
        self._hf_cache_dir = hf_cache_dir
        if isinstance(self._hf_cache_dir, str) and len(self._hf_cache_dir) > 0:
            try:
                Path(self._hf_cache_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # Load dataset (non-streaming)
        try:
            self._hf_ds = load_dataset(
                "ILSVRC/imagenet-1k",
                split=self.split,
                streaming=False,
                cache_dir=self._hf_cache_dir
            )
        except Exception as e:
            logger.error(f"Failed to load ILSVRC/imagenet-1k split='{self.split}': {e}")
            raise

        # Optionally disable HF image decoding to avoid PIL EXIF/XMP decode errors
        self._safe_decode = bool(safe_decode)
        if self._safe_decode:
            try:
                self._hf_ds = self._hf_ds.cast_column('image', HFImage(decode=False))
            except Exception:
                # If casting fails for any reason, fall back to default decoding
                self._safe_decode = False

        # Class names if available; otherwise numeric ids
        try:
            label_feat = self._hf_ds.features.get('label', None)
            self.classes = list(getattr(label_feat, 'names', [])) if label_feat is not None else list(range(1000))
        except Exception:
            self.classes = list(range(1000))

        # Optional class filtering: build index list only when requested
        self._indices: Optional[List[int]] = None
        target_idx: Optional[int] = None
        if self.class_subset is not None:
            try:
                if isinstance(self.class_subset, str):
                    if self.class_subset.isdigit():
                        target_idx = int(self.class_subset)
                    else:
                        if isinstance(self.classes[0], str):
                            name_to_idx = {n: i for i, n in enumerate(self.classes)}
                            target_idx = name_to_idx.get(self.class_subset, None)
                elif isinstance(self.class_subset, int):
                    target_idx = int(self.class_subset)
            except Exception:
                target_idx = None

            if target_idx is not None and 0 <= target_idx < len(self.classes):
                self._indices = []
                # Iterate once to collect indices; stop early if max_samples is set
                for i, ex in enumerate(self._hf_ds):
                    try:
                        if int(ex['label']) == target_idx:
                            self._indices.append(i)
                            if self._max_samples is not None and len(self._indices) >= self._max_samples:
                                break
                    except Exception:
                        continue
            # If unable to resolve subset, fall back to default behavior

        # Apply dataset-level truncation when no class subset indexing is used
        if self._indices is None and self._max_samples is not None:
            self._length = min(self._max_samples, len(self._hf_ds))
        else:
            self._length = len(self._indices) if self._indices is not None else len(self._hf_ds)

    def __len__(self) -> int:
        return int(self._length)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Resolve underlying index when class-subset is active
        if self._indices is not None:
            base_idx = self._indices[idx]
        else:
            base_idx = idx
        ex = self._hf_ds[int(base_idx)]

        try:
            if getattr(self, '_safe_decode', False):
                # ex['image'] is a dict with optional 'bytes' and 'path'
                img_info = ex.get('image')
                if isinstance(img_info, dict):
                    img_bytes = img_info.get('bytes', None)
                    if img_bytes is not None:
                        img_pil = Image.open(BytesIO(img_bytes))
                    else:
                        img_path = img_info.get('path')
                        img_pil = Image.open(img_path)
                else:
                    # Unexpected structure; fall back to array/PIL paths
                    img_pil = Image.fromarray(np.array(ex['image']))
            else:
                img_pil: Image.Image = ex['image'] if isinstance(ex['image'], Image.Image) else Image.fromarray(np.array(ex['image']))

            # Ensure RGB mode for consistency with class-conditional preprocessing
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
        except Exception:
            # Return a zero image on failure to decode
            img_tensor = torch.zeros((3, self.resolution, self.resolution), dtype=torch.uint8)
            label_tensor = torch.tensor(-1, dtype=torch.long)
            return {"image": img_tensor, "label": label_tensor}

        # Optional random horizontal flip for training split
        if self._flip_prob > 0.0 and random.random() < self._flip_prob:
            try:
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            except Exception:
                pass

        # Aspect-preserving resize + center crop to target resolution
        from src.utils.image import aspect_preserving_resize_and_center_crop
        img_pil = aspect_preserving_resize_and_center_crop(img_pil, self.resolution)
        img_np = np.array(img_pil, dtype=np.uint8)
        # Robust channel handling in case upstream decoders yield unexpected shapes
        if img_np.ndim == 2:
            # Grayscale -> RGB by channel stacking
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[-1] == 4:
            # Drop alpha channel
            img_np = img_np[..., :3]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

        try:
            lbl = int(ex['label'])
        except Exception:
            lbl = -1
        label_tensor = torch.tensor(lbl, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}

class TorchvisionCIFAR10(Dataset):
    """Wrapper around torchvision CIFAR-10 to return uint8 CHW images.

    Applies probabilistic horizontal flips during training according to
    `random_flip_prob` to match config-driven augmentation.
    """
    def __init__(self, split: str = 'train', download: bool = True, random_flip_prob: float = 0.0):
        super().__init__()
        train = (split == 'train')
        self.ds = torchvision.datasets.CIFAR10(
            root="./data/cifar10",
            train=train,
            download=download,
            transform=None,
            target_transform=None,
        )
        # Expose class names for downstream logging/sampling helpers
        try:
            self.classes = list(self.ds.classes)
        except Exception:
            self.classes = [str(i) for i in range(10)]
        self._flip_prob = float(random_flip_prob) if train else 0.0

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]
        # Apply random horizontal flip on the PIL image before tensor conversion
        if self._flip_prob > 0.0 and random.random() < self._flip_prob:
            try:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            except Exception:
                pass
        img_np = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {"image": img_tensor, "label": label_tensor}


_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


class KaggleImageFolderImagenet(Dataset):
    """Deprecated. Use TFDSImagenetResized64 instead."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError("KaggleImageFolderImagenet is deprecated. Use TFDSImagenetResized64 / imagenet64_tfds.")


class ImageNet21kFolder(Dataset):
    """Generic ImageNet-21k style folder loader with class subfolders."""

    def __init__(self, root_dir: str, split: str = 'train', resolution: int = 64, max_samples: Optional[int] = None, random_subset_seed: Optional[int] = None, class_subset: Optional[Union[int, str]] = None):
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
        self.class_subset = class_subset

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

        # Optional conditional class filtering before truncation
        if self.class_subset is not None and self.max_samples is not None:
            # Resolve class index by name or numeric id
            target_idx: Optional[int] = None
            try:
                if isinstance(self.class_subset, str):
                    if self.class_subset.isdigit():
                        target_idx = int(self.class_subset)
                    else:
                        target_idx = self.class_to_idx.get(self.class_subset, None)
                elif isinstance(self.class_subset, int):
                    target_idx = int(self.class_subset)
            except Exception:
                target_idx = None
            if target_idx is not None and 0 <= target_idx < len(self.classes):
                total_in_class = sum(1 for _, cls_idx in self.samples if cls_idx == target_idx)
                if total_in_class < int(self.max_samples):
                    self.samples = [(p, c) for (p, c) in self.samples if c == target_idx]

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
    if str(dataset_choice).lower() == 'imagenet64_tfds':
        H, W = tuple(getattr(config, 'input_size'))
        res = int(H)
        if res != 64 or res != int(W):
            raise ValueError("imagenet64_tfds requires input_size [64, 64]")
        dataset = TFDSImagenetResized64(
            split='train',
            max_samples=getattr(config, 'max_samples', None),
            class_subset=getattr(config, 'class_subset', None)
        )
        val_dataset = TFDSImagenetResized64(
            split='validation',
            max_samples=getattr(config, 'max_samples', None),
            class_subset=getattr(config, 'class_subset', None)
        )
    elif str(dataset_choice).lower() == 'imagenet21k_folder':
        root = getattr(config, 'imagenet21k_root', None)
        if not root:
            raise ValueError("--imagenet21k_root must be provided for imagenet21k_folder dataset")
        H, W = tuple(getattr(config, 'input_size'))
        res = int(H)
        dataset = ImageNet21kFolder(
            root_dir=root,
            split='train',
            resolution=res,
            max_samples=getattr(config, 'max_samples', None),
            class_subset=getattr(config, 'class_subset', None)
        )
        val_dataset = ImageNet21kFolder(
            root_dir=root,
            split='val',
            resolution=res,
            max_samples=getattr(config, 'max_samples', None),
            class_subset=getattr(config, 'class_subset', None)
        )
    elif str(dataset_choice).lower() == 'cifar10':
        flip_prob = float(getattr(config, 'random_flip_prob', 0.0))
        dataset = TorchvisionCIFAR10(split='train', download=True, random_flip_prob=flip_prob)
        val_dataset = TorchvisionCIFAR10(split='test', download=True, random_flip_prob=0.0)
    elif str(dataset_choice).lower() == 'imagenet1k_hf':
        # Use HF ILSVRC/imagenet-1k with user-specified resolution
        H, W = tuple(getattr(config, 'input_size'))
        res = int(H)
        if res != int(W):
            raise ValueError("imagenet1k_hf requires square input_size [H, W]")
        flip_prob = float(getattr(config, 'random_flip_prob', 0.0))
        safe_decode = bool(getattr(config, 'hf_safe_image_decode', True))
        dataset = HFImagenet1k(
            split='train',
            resolution=res,
            max_samples=getattr(config, 'max_samples', None),
            class_subset=getattr(config, 'class_subset', None),
            random_flip_prob=flip_prob,
            safe_decode=safe_decode,
        )
        val_dataset = HFImagenet1k(
            split='validation',
            resolution=res,
            max_samples=getattr(config, 'max_samples', None),
            class_subset=getattr(config, 'class_subset', None),
            random_flip_prob=0.0,
            safe_decode=safe_decode,
        )
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