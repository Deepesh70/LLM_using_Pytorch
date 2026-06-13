"""A small GPT-style language model.

Model flow:
1. Input token IDs arrive with shape [batch_size, sequence_length].
2. Token embeddings convert each token ID into a learned vector.
3. Positional embeddings add information about where each token appears.
4. Transformer blocks repeatedly mix information from previous tokens.
5. A final LayerNorm stabilizes the hidden states.
6. The output head projects each hidden state to vocabulary logits.

The model predicts the next token at every sequence position.
"""

import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """Minimal GPT model built from embeddings and transformer blocks."""

    def __init__(self, vocab_size, embedding_dim, context_length, drop_rate, num_heads, num_layers):
        super().__init__()

        # Token embedding table:
        # maps token IDs like 1212 or 318 into dense vectors of size embedding_dim.
        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)

        # Positional embedding table:
        # gives the model a learned vector for position 0, position 1, etc.
        # This is needed because attention alone does not know token order.
        self.pos_emb = nn.Embedding(context_length, embedding_dim)

        # Dropout regularizes training. It is automatically disabled by
        # model.eval() during generation.
        self.drop_emb = nn.Dropout(drop_rate)

        # Stack transformer blocks. Each block contains:
        # - masked multi-head self-attention
        # - feed-forward network
        # - residual connections
        # - layer normalization
        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(
                d_in=embedding_dim,
                context_length=context_length,
                dropout=drop_rate,
                num_heads=num_heads,
            ) for _ in range(num_layers)
        ])

        # Final normalization before converting hidden states into token scores.
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Language-model head:
        # converts each hidden vector into one score per vocabulary token.
        # Output shape becomes [batch_size, sequence_length, vocab_size].
        self.out_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        """Run a forward pass through the GPT model.

        Args:
            in_idx: Integer token IDs with shape [batch_size, sequence_length].

        Returns:
            Vocabulary logits with shape [batch_size, sequence_length, vocab_size].
        """

        batch_size, seq_len = in_idx.shape

        # Convert token IDs into vectors:
        # [batch_size, sequence_length] -> [batch_size, sequence_length, embedding_dim]
        tok_embeds = self.tok_emb(in_idx)

        # Create positions [0, 1, 2, ..., seq_len - 1] on the same device as input.
        # pos_embeds shape: [sequence_length, embedding_dim]
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)

        # Add token meaning and token position together. PyTorch broadcasts
        # pos_embeds across the batch dimension.
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # Each transformer block lets every token look back at earlier tokens
        # and refine its hidden representation.
        x = self.trf_blocks(x)

        x = self.final_norm(x)

        # Convert hidden states into raw token scores. Softmax is not applied
        # here because CrossEntropyLoss expects raw logits during training, and
        # argmax/sampling can work directly from logits during generation.
        logits = self.out_head(x)

        return logits


if __name__ == "__main__":
    # Quick standalone shape test for this file.
    VOCAB_SIZE = 50257
    EMBEDDING_DIM = 256
    CONTEXT_LENGTH = 1024
    NUM_HEADS = 8
    NUM_LAYERS = 4

    dummy_inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    print("Input Token IDs Shape:", dummy_inputs.shape)

    model = GPTModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_length=CONTEXT_LENGTH,
        drop_rate=0.1,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    )

    logits = model(dummy_inputs)
    print("Output Logits Shape:", logits.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")
