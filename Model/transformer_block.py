import torch
import torch.nn as nn

from multi_head import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_in, dropout):
        super().__init__()

        self.net = nn.Sequential(       # 1. Expand to 4x the dimension
            nn.Linear(d_in, 4*d_in),
            nn.GELU(),
            nn.Linear(4*d_in, d_in),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_in, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        #1. First layerNorm and MHA
        self.norm1 = nn.LayerNorm(d_in)
        self.attn = MultiHeadAttention(
            d_in = d_in ,
            d_out = d_in,
            context_length= context_length,
            dropout=dropout,
            num_heads= num_heads,
            qkv_bias=qkv_bias
        )

        #2. Second layerNorm and FFN
        self.norm2 = nn.LayerNorm(d_in)
        self.ff = FeedForward(d_in, dropout)

    def forward(self, x):
        # RESIDUAL CONNECTION 1: Add original x to the Attention output
        # Notice Pre-LayerNorm: x passes through norm1 BEFORE attention
        x = x + self.attn(self.norm1(x))
        # RESIDUAL CONNECTION 2: Add the new x to the FFN output
        # Notice Pre-LayerNorm: x passes through norm2 BEFORE the FFN
        x = x + self.ff(self.norm2(x))
        
        return x




if __name__ == "__main__":
    batch_size = 4
    context_length = 8
    embedding_dim = 256
    
    # 1. Simulate the embedded inputs
    embedded_inputs = torch.rand(batch_size, context_length, embedding_dim)
    print("Input shape to Transformer Block:", embedded_inputs.shape)
    
    # 2. Initialize the complete block
    block = TransformerBlock(
        d_in=embedding_dim, 
        context_length=1024, 
        dropout=0.1, 
        num_heads=8
    )
    
    # 3. Pass data through the layer
    output_vectors = block(embedded_inputs)
    
    print("Output shape from Transformer Block:", output_vectors.shape)