"""Transformer block used by the GPT model.

Block flow:
1. Normalize the input hidden states.
2. Apply masked multi-head self-attention so tokens can read earlier tokens.
3. Add the attention result back to the original input with a residual connection.
4. Normalize again.
5. Apply a feed-forward network to each token independently.
6. Add the feed-forward result back with another residual connection.

The block keeps the same tensor shape from input to output:
[batch_size, sequence_length, embedding_dim].
"""

import torch
import torch.nn as nn

from multi_head import MultiHeadAttention


class FeedForward(nn.Module):
    """Per-token MLP used after attention.

    Attention mixes information across tokens. This feed-forward network then
    transforms each token's vector independently to increase model capacity.
    """

    def __init__(self, d_in, dropout):
        super().__init__()

        self.net = nn.Sequential(
            # Expand hidden dimension, apply nonlinearity, then project back.
            nn.Linear(d_in, 4 * d_in),
            nn.GELU(),
            nn.Linear(4 * d_in, d_in),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One GPT transformer block with pre-layernorm."""

    def __init__(self, d_in, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Pre-layernorm before attention helps training stability.
        self.norm1 = nn.LayerNorm(d_in)
        self.attn = MultiHeadAttention(
            d_in=d_in,
            d_out=d_in,
            context_length=context_length,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        # Pre-layernorm before the feed-forward network.
        self.norm2 = nn.LayerNorm(d_in)
        self.ff = FeedForward(d_in, dropout)

    def forward(self, x):
        """Process hidden states through attention and feed-forward layers."""

        # Residual connection 1:
        # keep original x and add the attention update to it.
        x = x + self.attn(self.norm1(x))

        # Residual connection 2:
        # keep the attention-enhanced x and add the feed-forward update.
        x = x + self.ff(self.norm2(x))

        return x


if __name__ == "__main__":
    # Quick standalone shape test for this file.
    batch_size = 4
    context_length = 8
    embedding_dim = 256

    embedded_inputs = torch.rand(batch_size, context_length, embedding_dim)
    print("Input shape to Transformer Block:", embedded_inputs.shape)

    block = TransformerBlock(
        d_in=embedding_dim,
        context_length=1024,
        dropout=0.1,
        num_heads=8,
    )

    output_vectors = block(embedded_inputs)
    print("Output shape from Transformer Block:", output_vectors.shape)
