#!/usr/bin/env python3

import urllib.request
import os
from pathlib import Path

def download_sentencepiece_model():
    """Download the SentencePiece model from Google Storage"""
    
    url = "https://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model"
    local_path = "./sentencepiece.model"
    
    if os.path.exists(local_path):
        print(f"SentencePiece model already exists at {local_path}")
        return local_path
    
    print(f"Downloading SentencePiece model from {url}")
    print(f"Saving to {local_path}")
    
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"✅ Successfully downloaded SentencePiece model to {local_path}")
        
        # Verify file size
        file_size = os.path.getsize(local_path)
        print(f"File size: {file_size / 1024:.1f} KB")
        
        return local_path
        
    except Exception as e:
        print(f"❌ Failed to download SentencePiece model: {e}")
        raise e

if __name__ == "__main__":
    download_sentencepiece_model() 