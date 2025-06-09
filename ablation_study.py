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
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss, nll, perplexity = compute_metrics(logits, input_ids)
        
        # Backward pass
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
            
            logits = model(input_ids, attention_mask)
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

def run_ablation_study(config):
    """Run ablation study with given configuration."""
    # Initialize wandb
    wandb.init(
        project="gemma-ablation",
        config=config,
        name=f"{config['model_type']}-{config.get('model_size', '')}-{config.get('pe_type', '')}-{config.get('activation', '')}"
    )
    
    # Create wandb tables for metrics
    train_metrics_table = wandb.Table(columns=["epoch", "loss", "nll", "perplexity", "grad_norm"])
    val_metrics_table = wandb.Table(columns=["epoch", "loss", "nll", "perplexity"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets and dataloaders
    train_dataset = TinyStoriesDataset(max_text_len=config["max_seq_len"], split="train")
    val_dataset = TinyStoriesDataset(max_text_len=config["max_seq_len"], split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
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
    best_val_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, config["grad_clip"]
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log metrics to wandb tables
        train_metrics_table.add_data(
            epoch,
            train_metrics["train/loss"],
            train_metrics["train/nll"],
            train_metrics["train/perplexity"],
            train_metrics["train/grad_norm"]
        )
        
        val_metrics_table.add_data(
            epoch,
            val_metrics["val/loss"],
            val_metrics["val/nll"],
            val_metrics["val/perplexity"]
        )
        
        # Log metrics for wandb plots
        metrics = {
            **train_metrics,
            **val_metrics,
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0]
        }
        wandb.log(metrics)
        
        # Create and log plots
        wandb.log({
            "train_metrics": wandb.plot.line_series(
                xs=list(range(epoch + 1)),
                ys=[
                    [m["train/loss"] for m in train_metrics_table.data],
                    [m["train/nll"] for m in train_metrics_table.data],
                    [m["train/perplexity"] for m in train_metrics_table.data],
                    [m["train/grad_norm"] for m in train_metrics_table.data]
                ],
                keys=["Loss", "NLL", "Perplexity", "Grad Norm"],
                title="Training Metrics",
                xname="Epoch"
            ),
            "val_metrics": wandb.plot.line_series(
                xs=list(range(epoch + 1)),
                ys=[
                    [m["val/loss"] for m in val_metrics_table.data],
                    [m["val/nll"] for m in val_metrics_table.data],
                    [m["val/perplexity"] for m in val_metrics_table.data]
                ],
                keys=["Loss", "NLL", "Perplexity"],
                title="Validation Metrics",
                xname="Epoch"
            )
        })
        
        # Save best model
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            torch.save(model.state_dict(), f"best_model_{wandb.run.name}.pt")
    
    # Log final tables
    wandb.log({
        "train_metrics_table": train_metrics_table,
        "val_metrics_table": val_metrics_table
    })
    
    wandb.finish()

if __name__ == "__main__":
    # Base configuration
    base_config = {
        "max_seq_len": 2048,
        "batch_size": 4,
        "num_workers": 4,
        "learning_rate": 1e-4,
        "min_lr": 1e-5,
        "weight_decay": 0.01,
        "num_epochs": 10,
        "dropout": 0.1,
        "grad_clip": 1.0
    }
    
    # Ablation configurations
    ablation_configs = [
        # Baseline: Original transformer
        {
            **base_config,
            "model_type": "transformer",
            "d_model": 2048,
            "n_heads": 8,
            "n_layers": 18,
            "d_ff": 32768
        },
        
        # Model size ablation
        {**base_config, "model_type": "gemma", "model_size": "2b", "pe_type": "rope", "activation": "gelu"},
        {**base_config, "model_type": "gemma", "model_size": "7b", "pe_type": "rope", "activation": "gelu"},
        
        # Positional encoding ablation
        {**base_config, "model_type": "gemma", "model_size": "2b", "pe_type": "abs", "activation": "gelu"},
        {**base_config, "model_type": "gemma", "model_size": "2b", "pe_type": None, "activation": "gelu"},
        
        # Activation function ablation
        {**base_config, "model_type": "gemma", "model_size": "2b", "pe_type": "rope", "activation": "relu"},
        {**base_config, "model_type": "gemma", "model_size": "2b", "pe_type": "rope", "activation": "silu"}
    ]
    
    # Run ablation studies
    for config in ablation_configs:
        run_ablation_study(config)

