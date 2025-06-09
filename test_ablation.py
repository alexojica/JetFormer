import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import math
from tqdm import tqdm
from dataset import TinyStoriesDataset
from gemma_transformer import GemmaTransformer
from transformer import Transformer

def compute_metrics(logits, targets):
    """Compute loss, NLL, and perplexity metrics."""
    # Compute raw log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get the log probability of the correct class for each position
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Compute mean NLL (per token)
    mean_nll = nll.mean()
    
    # Compute cross entropy loss (includes reduction)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    # Compute perplexity
    perplexity = math.exp(mean_nll.item())
    
    return loss, mean_nll, perplexity

def train_epoch(model, train_loader, optimizer, scheduler, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_nll = 0
    total_perplexity = 0
    total_grad_norm = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        seq_len = input_ids.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool))
        padding_mask = attention_mask.unsqueeze(1)
        combined_mask = causal_mask.unsqueeze(0) & (padding_mask & padding_mask.transpose(-1, -2))
        combined_mask = combined_mask.unsqueeze(1)
        
        logits = model(input_ids, combined_mask)
        loss, nll, perplexity = compute_metrics(logits, input_ids)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        total_nll += nll
        total_perplexity += perplexity
        total_grad_norm += grad_norm.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "perplexity": f"{perplexity:.4f}",
            "grad_norm": f"{grad_norm.item():.4f}"
        })
    
    return {
        "train/loss": total_loss / num_batches,
        "train/nll": total_nll / num_batches,
        "train/perplexity": total_perplexity / num_batches,
        "train/grad_norm": total_grad_norm / num_batches
    }

def evaluate(model, val_loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_nll = 0
    total_perplexity = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            seq_len = input_ids.shape[1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool))
            padding_mask = attention_mask.unsqueeze(1)
            combined_mask = causal_mask.unsqueeze(0) & (padding_mask & padding_mask.transpose(-1, -2))
            combined_mask = combined_mask.unsqueeze(1)
            logits = model(input_ids, combined_mask)
            loss, nll, perplexity = compute_metrics(logits, input_ids)
            
            total_loss += loss.item()
            total_nll += nll
            total_perplexity += perplexity
            num_batches += 1
    
    return {
        "val/loss": total_loss / num_batches,
        "val/nll": total_nll / num_batches,
        "val/perplexity": total_perplexity / num_batches
    }

def create_model(config):
    """Create model based on configuration."""
    if config["model_type"] == "transformer":
        return Transformer(
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            max_seq_len=config["max_seq_len"]
        )
    else:  # gemma
        if config["model_size"] == "2b":
            return GemmaTransformer.from_2b_config(
                dropout=config["dropout"],
                max_seq_len=config["max_seq_len"],
                pe_type=config["pe_type"],
                activation=config["activation"]
            )
        else:  # 7b
            return GemmaTransformer.from_7b_config(
                dropout=config["dropout"],
                max_seq_len=config["max_seq_len"],
                pe_type=config["pe_type"],
                activation=config["activation"]
            )

def run_quick_test(config):
    """Run a quick test of the ablation study."""
    # Initialize wandb
    wandb.init(
        project="gemma-ablation-test",
        config=config,
        name=f"test-{config['model_type']}-{config.get('model_size', '')}-{config.get('pe_type', '')}-{config.get('activation', '')}"
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets and dataloaders with small subset
    train_dataset = TinyStoriesDataset(max_text_len=config["max_seq_len"], split="train", max_samples=100)
    val_dataset = TinyStoriesDataset(max_text_len=config["max_seq_len"], split="validation", max_samples=20)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0  # Single worker for quick testing
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = create_model(config)
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"] * len(train_loader),
        eta_min=config["min_lr"]
    )
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, config["grad_clip"]
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log metrics
        metrics = {
            **train_metrics,
            **val_metrics,
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0]
        }
        wandb.log(metrics)
    
    wandb.finish()

if __name__ == "__main__":
    # Quick test configuration
    test_config = {
        "max_seq_len": 128,  # Shorter sequence length
        "batch_size": 2,     # Smaller batch size
        "learning_rate": 1e-4,
        "min_lr": 1e-5,
        "weight_decay": 0.01,
        "num_epochs": 2,     # Fewer epochs
        "dropout": 0.1,
        "grad_clip": 1.0,
        "model_type": "gemma",
        "model_size": "2b",
        "pe_type": "rope",
        "activation": "gelu",
        "d_model": 256,      # Smaller model
        "n_heads": 4,        # Fewer heads
        "n_layers": 2,       # Fewer layers
        "d_ff": 1024         # Smaller feed-forward
    }
    
    # Run quick test
    run_quick_test(test_config) 