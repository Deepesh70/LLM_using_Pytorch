"""Masked multi-head self-attention used inside each transformer block.

Attention flow:
1. The same input sequence is projected into queries, keys, and values.
2. The embedding dimension is split into multiple smaller attention heads.
3. Each token compares its query with all key vectors to produce attention scores.
4. A causal mask blocks attention to future tokens.
5. Softmax converts scores into attention weights.
6. Weights are used to mix value vectors.
7. All heads are merged back into one embedding-sized vector per token.
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    "Self-attention" means queries, keys, and values all come from the same
    input sequence. "Causal" means token t can only attend to tokens <= t.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # The output dimension must split evenly across attention heads.
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Learned projections from input vectors to Q, K, and V vectors.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final learned projection after all heads are concatenated.
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Upper-triangular causal mask:
        # position i cannot see positions greater than i.
        # register_buffer stores it with the module but does not train it.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        """Apply masked multi-head attention.

        Args:
            x: Hidden states with shape [batch_size, num_tokens, d_in].

        Returns:
            Context vectors with shape [batch_size, num_tokens, d_out].
        """

        b, num_tokens, d_in = x.shape

        # Project input into keys, queries, and values.
        # Each has shape [batch_size, num_tokens, d_out].
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split d_out into num_heads separate heads.
        # [batch, tokens, d_out] -> [batch, tokens, heads, head_dim]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Move the head dimension before the token dimension so each head can
        # compute attention independently.
        # [batch, tokens, heads, head_dim] -> [batch, heads, tokens, head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Dot product attention scores:
        # [batch, heads, tokens, head_dim] @ [batch, heads, head_dim, tokens]
        # -> [batch, heads, tokens, tokens]
        attn_scores = queries @ keys.transpose(2, 3)

        # Mask out future-token scores before softmax. Those positions become
        # zero probability after softmax, so the model cannot cheat by seeing
        # future tokens during training.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale by sqrt(head_dim) to keep dot products numerically stable.
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values:
        # [batch, heads, tokens, tokens] @ [batch, heads, tokens, head_dim]
        # -> [batch, heads, tokens, head_dim]
        context_vec = attn_weights @ values

        # Merge all heads back into one vector per token.
        # [batch, heads, tokens, head_dim] -> [batch, tokens, heads, head_dim]
        # -> [batch, tokens, d_out]
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)

        return context_vec


if __name__ == "__main__":
    # Quick standalone shape test for this file.
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
        num_heads=8,
    )

    context_vectors = mha(embedded_inputs)
    print("Output shape from MHA:", context_vectors.shape)
