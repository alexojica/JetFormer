import torch
import torch.nn as nn
import math
from transformers import GPT2Tokenizer
from dataset import prepare_dataloaders
from torch.utils.data import Dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(context)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self attention with residual connection and layer norm
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_seq_len)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights
        self.output.weight = self.token_embedding.weight
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Get token embeddings
        x = self.token_embedding(x)
        
        # Add positional embeddings
        x = self.position_embedding(x)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final layer norm and output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return logits

class TinyStoriesDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        story = self.dataset[idx]['story']
        
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

def create_model(vocab_size=None):
    """Create a TinyStories transformer model."""
    if vocab_size is None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        vocab_size = tokenizer.vocab_size
    
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )
    
    return model

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate text using the trained model."""
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test the model and dataset
    train_loader, val_loader, tokenizer = prepare_dataloaders()
    model = create_model()
    
    # Print model summary
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass with a batch from the dataset
    batch = next(iter(train_loader))
    input_ids = batch['input_ids']
    
    output = model(input_ids)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test generation
    prompt = "Once upon a time"
    generated = generate_text(model, tokenizer, prompt)
    print(f"\nGenerated text:")
    print(generated) 