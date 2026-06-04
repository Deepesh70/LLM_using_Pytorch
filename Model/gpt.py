import torch
import torch.nn as nn
from transformer_block import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, drop_rate, num_heads, num_layers):
        super().__init__()

        #1. Embeddings 

        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(context_length, embedding_dim)
        self.drop_emb = nn.Dropout(drop_rate)

        #2. Stacking Transformer Blocks
        #nn.Sequential allows us to easily stack multiple blocks together

        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(
                d_in=embedding_dim,
                context_length=context_length,
                dropout = drop_rate,
                num_heads = num_heads
            ) for _ in range(num_layers)
        ])

        #3 Final stabilization layer
        self.final_norm = nn.LayerNorm(embedding_dim)

        #4 Language Model Head
        #project from embedding space back to vocab space
        self.out_head =nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # get embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)

        x= tok_embeds + pos_embeds
        x = self.drop_emb(x)

        #pass through transformer blocks
        x = self.trf_blocks(x)

        #Final Norm
        x = self.final_norm(x)

        #convert context vectors to vocab predictions/logits

        prediction = self.out_head(x)

        return prediction
    

if __name__ == "__main__":
    # Hyperparameters for a small prototype model
    VOCAB_SIZE = 50257
    EMBEDDING_DIM = 256
    CONTEXT_LENGTH = 1024
    NUM_HEADS = 8
    NUM_LAYERS = 4 # We will stack 4 Transformer Blocks
    
    # 1. Simulate the input from your DataLoader (Step 1)
    # Shape: [Batch=2, Tokens=8]
    dummy_inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    
    print("Input Token IDs Shape:", dummy_inputs.shape)
    
    # 2. Initialize the entire GPT Model
    model = GPTModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_length=CONTEXT_LENGTH,
        drop_rate=0.1,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )
    
    # 3. Perform a forward pass
    logits = model(dummy_inputs)
    
    print("Output Logits Shape:", logits.shape)
    
    # 4. Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")