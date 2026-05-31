import torch
import torch.nn as nn

from dataset_prep import create_dataloader_v1

vocab_size = 50257  # GPT-2's vocabulary size
embedding_dim = 256 
context_length = 1024

class EmbeddingsLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):

        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        
        self.pos_emb = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape

        tok_embeds = self.token_emb(input_ids)


        pos_indices = torch.arange(seq_length, device=input_ids.device)

        pos_embeds = self.pos_emb(pos_indices)

        x = tok_embeds + pos_embeds
        return x

if __name__ == "__main__":

    raw_text = "This is a simple test text to verify that the PyTorch dataloader is functioning exactly as intended." * 100

    dataloader = create_dataloader_v1(
        raw_text, 
        batch_size=4, 
        max_length=8, 
        stride=4, 
        shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)


    print("Original Inputs Shape:", inputs.shape)
    
    # Initialize the embedding layer
    embedding_layer = EmbeddingsLayer(vocab_size, embedding_dim, context_length)
    
    # Pass the inputs through
    embedded_inputs = embedding_layer(inputs)
    
    # Verify the final 3D tensor
    print("Embedded Inputs Shape:", embedded_inputs.shape)