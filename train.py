import torch
import torch.nn as nn
from torch.optim import AdamW
import time
from tqdm import tqdm
from dataset import prepare_dataloaders
from transformer import DecoderOnlyTransformer
import math
import os
import wandb

def get_grad_norm(model):
    """Calculate the total gradient norm of the model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm)

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=3,
    learning_rate=3e-4,
    weight_decay=0.1,
    max_grad_norm=1.0,
    device='cpu',
    save_path='best_model.pt',
    use_wandb=True,
    wandb_project="tinystories-transformer",
    wandb_name=None
):
    """Train the model."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token
    best_val_loss = float('inf')
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "architecture": "decoder-only-transformer",
                "dataset": "TinyStories",
                "d_model": model.token_embedding.embedding_dim,
                "num_heads": model.decoder_blocks[0].self_attn.num_heads,
                "num_layers": len(model.decoder_blocks),
                "d_ff": model.decoder_blocks[0].feed_forward.net[0].out_features,
                "max_seq_len": model.position_embedding.pe.size(1),
                "vocab_size": model.token_embedding.num_embeddings,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "batch_size": train_loader.batch_size,
                "num_epochs": num_epochs,
                "device": str(device),
                "num_parameters": sum(p.numel() for p in model.parameters())
            }
        )
        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)
    
    print(f"\nTraining on {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Max gradient norm: {max_grad_norm}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Get gradient norm before step
            grad_norm = get_grad_norm(model)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_tokens += input_ids.numel()
            batch_count += 1
            
            # Calculate tokens per second
            elapsed_time = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed_time
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/batch_count:.4f}',
                'grad_norm': f'{grad_norm:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'tokens/sec': f'{tokens_per_sec:.0f}'
            })
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/avg_loss': total_loss/batch_count,
                    'train/grad_norm': grad_norm,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/epoch': epoch + 1,
                    'train/step': batch_count
                })
        
        # Validation
        model.eval()
        val_loss = 0
        val_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                
                val_loss += loss.item()
                val_tokens += input_ids.numel()
        
        val_loss = val_loss / len(val_loader)
        val_perplexity = math.exp(val_loss)
        
        # Log validation metrics to wandb
        if use_wandb:
            wandb.log({
                'val/loss': val_loss,
                'val/perplexity': val_perplexity,
                'val/epoch': epoch + 1
            })
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Training loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation perplexity: {val_perplexity:.2f}")
        print(f"Training speed: {tokens_per_sec:.0f} tokens/sec")
        print(f"Epoch time: {time.time() - start_time:.2f} seconds")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_perplexity': val_perplexity
            }
            torch.save(checkpoint, save_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
            
            # Save model to wandb
            if use_wandb:
                wandb.save(save_path)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")
    
    if use_wandb:
        wandb.finish()

def main():
    # Model parameters
    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 1024
    max_seq_len = 512
    vocab_size = 50257  # GPT-2 vocabulary size
    
    # Training parameters
    batch_size = 32
    num_epochs = 3
    learning_rate = 3e-4
    weight_decay = 0.1
    max_grad_norm = 1.0
    subset_size = None  # Number of training samples to use
    
    # Wandb parameters
    use_wandb = True
    wandb_project = "tinystories-transformer"
    wandb_name = f"transformer-d{d_model}-h{num_heads}-l{num_layers}"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare dataloaders
    train_loader, val_loader, tokenizer = prepare_dataloaders(
        batch_size=batch_size,
        max_length=max_seq_len,
        seed=42,
        subset_size=subset_size
    )
    
    # Create model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        save_path='best_model.pt',
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_name=wandb_name
    )

if __name__ == "__main__":
    main()