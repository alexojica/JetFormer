import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import numpy as np

class TinyStoriesDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        story = self.dataset[idx]['text']
        
        # Tokenize the story
        encoding = self.tokenizer(
            story,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding

def prepare_dataloaders(batch_size=32, max_length=512, test=False, seed=None, subset_size=None):
    """Prepare TinyStories dataset and create dataloaders."""
    from datasets import load_dataset
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Load dataset
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    if test:
        # For test data, use the validation split
        print("\nPreparing test dataset...")
        test_dataset = TinyStoriesDataset(dataset['validation'], tokenizer, max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        return test_loader, None, tokenizer
    
    else:
        # For training/validation data
        print("\nPreparing training and validation datasets...")
        train_dataset = TinyStoriesDataset(dataset['train'], tokenizer, max_length)
        val_dataset = TinyStoriesDataset(dataset['validation'], tokenizer, max_length)
        
        # If subset_size is provided, limit the training dataset
        if subset_size is not None:
            print(f"\nUsing subset of {subset_size} training samples")
            indices = np.random.permutation(len(train_dataset))[:subset_size]
            train_dataset = torch.utils.data.Subset(train_dataset, indices.tolist())
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\nDataset prepared:")
        print(f"- Training samples: {len(train_dataset)}")
        print(f"- Validation samples: {len(val_dataset)}")
        print(f"- Batch size: {batch_size}")
        print(f"- Max sequence length: {max_length}")
        
        return train_loader, val_loader, tokenizer