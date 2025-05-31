import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict, Any, List
import json
from pathlib import Path
import wandb
import time
import os
import requests
from io import BytesIO
import hashlib
from datasets import load_dataset
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
from sentencepiece import SentencePieceProcessor

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

        # Download/prepare tokenizer model
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
        
        print("Loading LAION-POP dataset...")
        self.data = self._load_laion_pop_dataset()
        print(f"Loaded {len(self.data)} samples from LAION-POP dataset")
        
        self._predownload_images()
    
    def _load_laion_pop_dataset(self):
        """Load LAION-POP dataset from Hugging Face"""
        try:
            print("Downloading LAION-POP dataset from Hugging Face...")
            
            # Load the full dataset - LAION-POP is relatively small (600k samples)
            dataset = load_dataset("laion/laion-pop", split="train", streaming=False)
            
            print(f"Raw dataset size: {len(dataset)}")
            
            # Filter and collect samples
            data = []
            count = 0
            filtered_count = 0
            
            print("Filtering LAION-POP samples...")
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
                        print(f"Processed {count + filtered_count} samples, kept {count}, filtered {filtered_count}")
                        
                except Exception as e:
                    filtered_count += 1
                    continue
            
            print(f"Filtering complete: {count} samples kept, {filtered_count} filtered out")
            return data
            
        except Exception as e:
            print(f"Error loading LAION-POP dataset: {e}")
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
                except:
                    pass
                
                return image
                
            except Exception as e:
                if attempt == max_retries - 1:
                    pass
                time.sleep(0.5)
        
        return None
    
    def _predownload_images(self, num_predownload=200):
        """Pre-download some images for faster training startup"""
        print(f"Pre-downloading {num_predownload} images...")
        
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
            print(image)
            return image is not None
        
        with ThreadPoolExecutor(max_workers=min(8, self.num_workers)) as executor:
            futures = [executor.submit(download_worker, i) for i in range(min(num_predownload, len(self.data)))]
            
            success_count = 0
            for i, future in enumerate(as_completed(futures)):
                if future.result():
                    success_count += 1
                
                if (i + 1) % 50 == 0:
                    print(f"Pre-downloaded {i + 1}/{num_predownload} images ({success_count} successful)")
        
        print(f"Pre-download completed: {success_count}/{num_predownload} successful")
    
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
        
    
        text_mask = [1] * len(tokens)
        text_loss = [1] * len(tokens)
        pad_value = 1
        
        # Truncate if too long
        if len(tokens) > self.max_text_len:
            tokens = tokens[:self.max_text_len]
            text_mask = text_mask[:self.max_text_len]
            text_loss = text_loss[:self.max_text_len]
        else:
            # Pad to max_text_len
            padding_len = self.max_text_len - len(tokens)
            tokens.extend([pad_value] * padding_len)
            text_mask.extend([pad_value] * padding_len)
            text_loss.extend([pad_value] * padding_len)
        
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