import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        print(f"\n--- ATTENTION FORWARD PASS START ---")
        print(f"0. Input (x) shape: {x.shape} -> [Batch, SeqLen, EmbedDim]")

        # 1. Generate Q, K, V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        print(f"1. Keys shape:    {keys.shape}")
        print(f"   Queries shape: {queries.shape}")
        print(f"   Values shape:  {values.shape}")

        # 2. Compute Attention Scores (Q * K^T)
        print(f"   Transposed Keys shape: {keys.transpose(1, 2).shape} -> [Batch, EmbedDim, SeqLen]")
        attn_scores = queries @ keys.transpose(1, 2)
        print(f"2. Unmasked Attn Scores:  {attn_scores.shape} -> [Batch, SeqLen, SeqLen] (The 8x8 grid)")

        # 3. Apply the Causal Mask
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        print(f"3. Masked Attn Scores:    {attn_scores.shape} (Upper right triangle is now -inf)")

        # 4. Scale and Softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        print(f"4. Attn Weights (Softmax):{attn_weights.shape} (Rows now sum to 1.0)")
        
        attn_weights = self.dropout(attn_weights)

        # 5. Multiply by Values
        context_vec = attn_weights @ values
        print(f"5. Final Context Vector:  {context_vec.shape} -> [Batch, SeqLen, EmbedDim]")
        print(f"--- ATTENTION FORWARD PASS END ---\n")
        
        return context_vec

if __name__ == "__main__":
    # Simulate the embedded inputs from your previous step
    batch_size = 4
    context_length = 8
    embedding_dim = 256
    
    # Random tensor simulating your embedded data
    embedded_inputs = torch.rand(batch_size, context_length, embedding_dim)
    
    print("Input shape to Attention:", embedded_inputs.shape)
    
    # Initialize the Causal Attention module
    attention = CausalSelfAttention(
        d_in=embedding_dim, 
        d_out=embedding_dim, 
        context_length=1024, # Our absolute max context 
        dropout=0.1
    )
    
    # Pass data through the layer
    context_vectors = attention(embedded_inputs)
    
    print("Output shape from Attention:", context_vectors.shape)