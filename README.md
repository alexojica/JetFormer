# TinyStories Transformer

A PyTorch implementation of a transformer model trained on the TinyStories dataset.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:
```bash
python train.py
```

The script will:
- Download and prepare the TinyStories dataset
- Train the transformer model
- Save the best model checkpoint

### Testing

To test the model:
```bash
python test.py --eval  # Run evaluation
python test.py --generate --prompt "Once upon a time"  # Generate text
```

## Model Architecture

- Decoder-only transformer
- GPT2 pretrained tokenizer
- Causal self-attention
- Position-wise feed-forward networks
- Layer normalization

## Dataset

The model is trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), which contains:
- Simple, short stories
- Written in a way that's easy for language models to learn
- Suitable for training small language models
