"""
This module details the creation of an Embeddings Layer as used in Transformers (like GPT models).
It handles the mapping of discrete text tokens to high-dimensional continuous vectors,
and injects positional awareness into the tokens.
"""

import torch
import torch.nn as nn

from dataset_prep import create_dataloader_v1

# --- Global Configurations / Hyperparameters ---
# The total number of unique tokens the model can recognize (Matches GPT-2's tokenizer).
vocab_size = 50257  
# The size/dimension of the vector representation for each token and position.
embedding_dim = 256 
# The maximum sequence length (context window) that the model processes at one time.
context_length = 1024

class EmbeddingsLayer(nn.Module):
    """
    Combines Token Embeddings and Positional Embeddings to represent 
    text context for Transformer-based architectures.
    """
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()

        # Token Embedding: Maps discrete token IDs into a continuous vector space of size 'embedding_dim'.
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional Embedding: Maps position indices into the same continuous vector space.
        # This provides the transformer with information about the order of words in the sequence.
        self.pos_emb = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids):
        # input_ids shape is typically (batch_size, sequence_length)
        batch_size, seq_length = input_ids.shape

        # Passes token IDs into the token embedding layer to get semantic representations.
        # Shape becomes (batch_size, seq_length, embedding_dim)
        tok_embeds = self.token_emb(input_ids)

        # Generate a tensor of indices [0, 1, ..., seq_length - 1] to represent sequence positions.
        pos_indices = torch.arange(seq_length, device=input_ids.device)

        # Passes the position indices into the positional embedding layer.
        # Shape becomes (seq_length, embedding_dim)
        pos_embeds = self.pos_emb(pos_indices)

        # Add the token embeddings and positional embeddings together.
        # Through broadcasting, the (seq_length, embedding_dim) pos_embeds tensor 
        # is added to each sample in the (batch_size, seq_length, embedding_dim) tok_embeds tensor.
        x = tok_embeds + pos_embeds
        
        # Final output is a rich 3D tensor containing semantic and positional contexts.
        return x

if __name__ == "__main__":
    # --- Workflow Test / Demonstration ---

    # 1. Create a dummy corpus (raw text) to feed into our data pipeline.
    raw_text = "This is a simple test text to verify that the PyTorch dataloader is functioning exactly as intended." * 100

    # 2. Instantiate the Data Loader. 
    # This processes raw text into overlapping token sequences (inputs) and next-token targets.
    dataloader = create_dataloader_v1(
        raw_text, 
        batch_size=4,    # Number of sequences processed in parallel in one forward pass
        max_length=8,    # Sequence length (number of input tokens per sequence)
        stride=4,        # Sliding window overlap configuration
        shuffle=False    # Sequential reading
    )

    # 3. Pull the first batch from our dataset iterator
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    # 'inputs' is 2D: (batch_size=4, max_length=8)
    print("Original Inputs Shape:", inputs.shape)
    
    # 4. Initialize our custom combined Embedding layer
    embedding_layer = EmbeddingsLayer(vocab_size, embedding_dim, context_length)
    
    # 5. Pass the discrete tokenized inputs into the layer to add semantic and positional meaning
    embedded_inputs = embedding_layer(inputs)
    
    # 6. Verify the final structure. 
    # Shape increases from 2D token IDs (4x8) to a 3D tensor of embedded features (4x8x256).
    print("Embedded Inputs Shape:", embedded_inputs.shape)