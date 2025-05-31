import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import wandb
import argparse
import yaml
import math
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset import LAIONPOPTextImageDataset
from jetformer import JetFormer
from PIL import Image
import torchvision.transforms as transforms

IMAGE_SIZE = (256, 256, 3)

def image_loss_fn(gmm_dist, image_tokens, log_det, noise_nll=0):
    log_prob = -gmm_dist.log_prob(image_tokens)
    log_prob = log_prob.reshape(log_det.shape[0], -1)
    nll = (torch.sum(log_prob, dim=1) + noise_nll) / (IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2])
    nll = nll / math.log(2)
    log_det = log_det / (IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]) - math.log(127.5)
    log_det = log_det / math.log(2)
    return nll - log_det, log_det

def compute_text_loss(text_logits, text_tokens, text_loss_mask, vocab_size):
    logits_flat = text_logits.reshape(-1, vocab_size)  # [batch*seq_len, vocab_size]
    tokens_flat = text_tokens.reshape(-1)  # [batch*seq_len]
    mask_flat = text_loss_mask.reshape(-1)  # [batch*seq_len]
    
    if not mask_flat.any():
        return torch.tensor(0.0, device=text_logits.device, requires_grad=True)
    
    loss = F.cross_entropy(
        logits_flat[mask_flat],
        tokens_flat[mask_flat],
        reduction='mean'
    )
    return loss

def generate_text_to_image_samples(model, dataset, device, num_samples=3, temperature=1.0):
    model.eval()
    samples = []
    
    prompt_texts = [
        "a car",
        "a cat", 
        "a dog"
    ]
    with torch.no_grad():
        for i, prompt_text in enumerate(prompt_texts[:num_samples]):
            try:
                tokenized = dataset.tokenize_text(prompt_text)
                text_tokens = tokenized['tokens'].unsqueeze(0).to(device)  # [1, seq_len]
                text_mask = tokenized['text_mask'].unsqueeze(0).to(device)
                
                image_tokens = torch.zeros(1, model.image_seq_len, model.image_token_dim, device=device)
                
                text_first_mask = torch.tensor([True], device=device)
                
                total_len = text_tokens.shape[1] + model.image_seq_len + 1  # +1 for BOI token
                full_mask = torch.ones(1, text_tokens.shape[1], device=device, dtype=torch.bool)
                
                for pos in range(model.image_seq_len):
                    _, image_logits = model(text_tokens, image_tokens, text_first_mask, full_mask)
                    
                    if pos < image_logits.shape[1]:
                        gmm_dist, _ = model.gmm(image_logits[:, pos:pos+1], image_tokens[:, pos:pos+1])
                        
                        if temperature != 1.0:
                            sampled_token = gmm_dist.sample()
                            sampled_token = sampled_token * temperature
                        else:
                            sampled_token = gmm_dist.sample()
                        
                        image_tokens[0, pos] = sampled_token.squeeze()
                
                images, _ = model.jet.inverse(image_tokens)
                
                image = images[0]
                image = torch.clamp(image, -1, 1)
                image = (image + 1) / 2
                image_np = image.permute(1, 2, 0).cpu().numpy()
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                
                samples.append({
                    'prompt': prompt_text,
                    'image': image_pil
                })
                    
            except Exception as e:
                print(f"Failed to generate text-to-image sample {i}: {e}")
                import traceback
                traceback.print_exc()
                placeholder = Image.new('RGB', (256, 256), color='red')
                samples.append({
                    'prompt': prompt_text,
                    'image': placeholder
                })
    
    model.train()
    return samples

def train(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    
    wandb.init(
        project="jetformer-laion-pop",
        config=config_dict
    )
    if os.environ.get('DEBUG') is not None:
        torch.autograd.set_detect_anomaly(True)
    
    config = wandb.config
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    model = JetFormer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        num_mixtures=config.num_mixtures,
        dropout=config.dropout,
        jet_depth=config.jet_depth,
        jet_block_depth=config.jet_block_depth,
        jet_emb_dim=config.jet_emb_dim,
        jet_num_heads=config.jet_num_heads,
        patch_size=config.patch_size,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    jet_params = sum(p.numel() for p in model.jet.parameters())
    transformer_params = total_params - jet_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Jet flow parameters: {jet_params:,}")
    print(f"Transformer parameters: {transformer_params:,}")
    
    wandb.summary.update({
        "model/total_params": total_params,
        "model/jet_params": jet_params,
        "model/transformer_params": transformer_params
    })

    compiled_enabled = config.get('torch_compile', False)
    if compiled_enabled:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile")
    else:
        print("Model not compiled with torch.compile")
    
    print("Creating LAION-POP dataset...")
    dataset = LAIONPOPTextImageDataset(
        vocab_size=config.vocab_size,
        max_text_len=config.max_seq_len,
        max_samples=config.max_samples,
        use_cogvlm_captions=config.get('use_cogvlm_captions', True),
        min_resolution=config.get('min_resolution', 512),
        num_workers=config.get('num_workers', 4),
        ignore_pad=config.get('ignore_pad', False)
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=False,
        drop_last=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001, betas=(0.9, 0.95))
    
    total_steps = len(dataloader) * config.num_epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1, # warmup
        anneal_strategy='cos'
    )
    
    model.train()
    step = 0
    
    for epoch in range(config.num_epochs):
        epoch_losses = {
            'total': 0.0,
            'text': 0.0,
            'image_gen': 0.0,
            'flow': 0.0
        }
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            start_time = time.time()
            
            text_tokens = batch['text'].to(device)
            text_mask = batch['text_mask'].to(device)
            text_loss_mask = batch['text_loss'].to(device)
            images = batch['image'].to(device)
            batch_size = text_tokens.shape[0]

            log_det, image_tokens = model.flow(images)
            
            text_first_mask = torch.bernoulli(torch.ones(batch_size) * 0.5).bool()
            # for each sequence, choose a uniform noise from 0 to config.noise_std and consider text_first_mask
            noise_std = torch.rand(batch_size) * config.get('noise_std', 0.1)
            noise_std = torch.where(text_first_mask, noise_std, 0.0).unsqueeze(-1).unsqueeze(-1)
            
            # add noise to image tokens for text-first samples
            noise = torch.randn_like(image_tokens) * noise_std.to(device)
            image_tokens_noisy = image_tokens + noise
                            
            text_logits, image_logits = model(text_tokens, image_tokens_noisy, text_first_mask, text_mask)
            text_loss = compute_text_loss(text_logits, text_tokens, text_loss_mask, config.vocab_size)
            
            gmm_dist, image_tokens = model.gmm(image_logits, image_tokens)
            image_loss, log_det_loss = image_loss_fn(gmm_dist, image_tokens, log_det, 0.0)

            
            loss = (config.get('text_loss_weight', 1.0) * text_loss) + (config.get('image_loss_weight', 1.0) * image_loss.mean())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses['total'] += loss.item()
            epoch_losses['text'] += text_loss.item()
            epoch_losses['image_gen'] += image_loss.mean().item()
            epoch_losses['flow'] += log_det_loss.mean().item()
            num_batches += 1
            
            if step % 10 == 0:
                
                wandb.log({
                    "train/total_loss": loss.item(),
                    "train/text_loss": text_loss.item(),
                    "train/image_gen_loss": image_loss.mean().item(),
                    "train/flow_loss": log_det_loss.mean().item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/step": step,
                    "train/epoch": epoch,
                    "train/batch_time": time.time() - start_time,
                })
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{config.num_epochs}, "
                        f"Batch {batch_idx}/{len(dataloader)}, "
                        f"Total Loss: {loss.item():.4f}, "
                        f"Text: {text_loss.item():.4f}, "
                        f"Image Gen: {image_loss.mean().item():.4f}, "
                        f"Flow: {log_det_loss.mean().item():.4f}")
                
                print("Generating samples for wandb logging...")
                try:
                    text_to_image_samples = generate_text_to_image_samples(model, dataset, device, num_samples=3, temperature=0.8)
                    
                    # Create a new table each time instead of reusing the same one
                    generation_table = wandb.Table(
                        columns=["Batch", "Sample ID", "Text Prompt", "Image"]
                    )
                    
                    for i, sample in enumerate(text_to_image_samples):
                        generation_table.add_data(
                            batch_idx,
                            i+1,
                            sample['prompt'],
                            wandb.Image(sample['image'])
                        )
                    
                    # Also log individual images with step for better tracking
                    image_dict = {}
                    for i, sample in enumerate(text_to_image_samples):
                        image_dict[f"generation/image_{i+1}_{sample['prompt']}"] = wandb.Image(sample['image'])
                    
                    wandb.log({
                        "generation/samples_table": generation_table,
                        **image_dict,
                        "generation/step": step
                    })
                    
                    print(f"  Text-to-image samples: {len(text_to_image_samples)}")
                except Exception as e:
                    print(f"Failed to generate samples: {e}")
                    import traceback
                    traceback.print_exc()

            if batch_idx % 2000 == 0:
                print(f"Saving checkpoint for epoch {epoch+1} at batch {batch_idx}")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'config': dict(config),
                }
                torch.save(checkpoint, f'jetformer_laion_pop_epoch_{epoch+1}_batch_{batch_idx}.pt')
                print(f"✓ Saved checkpoint for epoch {epoch+1} at batch {batch_idx}")
                
            step += 1
                
    print("Training completed!")
    wandb.finish()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train JetFormer model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    # train model
    model = train(args.config)