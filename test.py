import torch
import torch.nn as nn
from tqdm import tqdm
import math
import argparse
from dataset import prepare_dataloaders
from transformer import TinyStoriesTransformer

def load_model(model_path, device):
    """Load the trained model."""
    # Model parameters (must match training configuration)
    d_model = 128
    num_heads = 4
    num_layers = 3
    d_ff = 512
    max_seq_len = 256
    vocab_size = 50257  # GPT-2 vocabulary size
    
    # Create model
    model = TinyStoriesTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0  # No dropout during evaluation
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    
    # Print checkpoint information if available
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    if 'val_perplexity' in checkpoint:
        print(f"Validation perplexity: {checkpoint['val_perplexity']:.2f}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch'] + 1} epochs")
    
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test set."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            correct = (predictions == input_ids).sum().item()
            total_correct += correct
            total_tokens += input_ids.numel()
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_tokens
    
    print("\nTest Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total Tokens: {total_tokens:,}")
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'total_tokens': total_tokens
    }

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, device='cpu'):
    """Generate text using the model."""
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    print(f"Input shape: {input_ids.shape}")
    
    # Generate text
    with torch.no_grad():
        for i in range(max_length):
            # Get model predictions
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Print token information for debugging
            if i < 5:  # Print first few tokens for debugging
                token = next_token.item()
                token_text = tokenizer.decode([token])
                print(f"Generated token {i}: {token} -> '{token_text}'")
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                print("Generated EOS token, stopping generation")
                break
    
    # Decode and return generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Final sequence length: {len(input_ids[0])}")
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Test the trained model')
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                      help='Path to the trained model')
    parser.add_argument('--eval', action='store_true',
                      help='Run evaluation on test set')
    parser.add_argument('--generate', action='store_true',
                      help='Generate text samples')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                      help='Prompt for text generation')
    parser.add_argument('--max_length', type=int, default=100,
                      help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for text generation')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    if args.eval:
        # Prepare test dataloader
        test_loader, _, tokenizer = prepare_dataloaders(
            batch_size=args.batch_size,
            max_length=256,
            test=True,
            seed=42
        )
        
        # Evaluate model
        metrics = evaluate_model(model, test_loader, device)
    
    if args.generate:
        # Prepare tokenizer
        _, _, tokenizer = prepare_dataloaders(
            batch_size=args.batch_size,
            max_length=256,
            test=True,
            seed=42
        )
        
        # Generate text
        print(f"\nGenerating text with prompt: '{args.prompt}'")
        print(f"Temperature: {args.temperature}")
        print(f"Max length: {args.max_length}")
        generated_text = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            device=device
        )
        print("\nGenerated text:")
        print(generated_text)

if __name__ == "__main__":
    main() 