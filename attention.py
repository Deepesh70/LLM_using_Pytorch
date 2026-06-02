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

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)

        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores/keys.shape[-1] **0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
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