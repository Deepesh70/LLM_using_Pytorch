import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # CORRECTED: Properly initialize all three linear layers
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 1. Project to Q, K, V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 2. THE CRITICAL MISSING STEP: SPLIT INTO HEADS
        # Reshape [batch, tokens, 256] -> [batch, tokens, 8, 32]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 3. Transpose to group the heads for parallel math
        # Reshape [batch, tokens, 8, 32] -> [batch, 8, tokens, 32]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4. Attention Scores
        # Now transpose(2,3) works because we actually have 4 dimensions
        attn_scores = queries @ keys.transpose(2, 3)

        # 5. Mask and Softmax
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. Multiply by values
        context_vec = attn_weights @ values
        
        # 7. Merge the heads back together
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # 8. Final projection
        context_vec = self.out_proj(context_vec)

        return context_vec



if __name__ == "__main__":
    batch_size = 4
    context_length = 8
    embedding_dim = 256
    
    embedded_inputs = torch.rand(batch_size, context_length, embedding_dim)
    print("Input shape to MHA:", embedded_inputs.shape)
    
    mha = MultiHeadAttention(
        d_in=embedding_dim, 
        d_out=embedding_dim, 
        context_length=1024, 
        dropout=0.1,
        num_heads=8 # Splitting 256 into 8 heads of 32
    )
    
    context_vectors = mha(embedded_inputs)
    print("Output shape from MHA:", context_vectors.shape)