import torch
import urllib.request
import os
from src.utils.logging import get_logger

logger = get_logger(__name__)

def download_sentencepiece_model():
    """Download the SentencePiece model from Google Storage"""
    url = "https://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model"
    local_path = "./sentencepiece.model"
    if os.path.exists(local_path):
        logger.info(f"SentencePiece model already exists at {local_path}")
        return local_path
    logger.info(f"Downloading SentencePiece model from {url} -> {local_path}")
    try:
        urllib.request.urlretrieve(url, local_path)
        file_size = os.path.getsize(local_path)
        logger.info(f"Successfully downloaded SentencePiece model to {local_path} ({file_size / 1024:.1f} KB)")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download SentencePiece model: {e}")
        raise e

if __name__ == "__main__":
    download_sentencepiece_model()
