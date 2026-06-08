import torch
import tiktoken
from gpt import GPTModel

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is the initial integer tensor of your prompt. Shape: [Batch, SeqLen]
    
    for _ in range(max_new_tokens):
        # 1. Crop context
        # If the generated text gets longer than what the model can handle (e.g., 1024),
        # we must chop off the oldest words so it fits in the context window.
        idx_cond = idx[:, -context_size:]
        
        # 2. Forward Pass
        # CRITICAL: torch.no_grad() tells PyTorch to turn off the calculus engine.
        # We are not training here. Tracking gradients during inference wastes massive memory.
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 3. Focus on the final step
        # Logits shape: [Batch, SeqLen, VocabSize]
        # We only care about the model's prediction for the very LAST word.
        logits = logits[:, -1, :] # Shape becomes: [Batch, VocabSize]
        
        # 4. Greedy Decoding
        # Find the index (token ID) with the absolute highest score
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) # Shape: [Batch, 1]
        
        # 5. Append to the sequence
        # We concatenate the new token to the end of our running sequence
        idx = torch.cat((idx, idx_next), dim=1) # Shape: [Batch, SeqLen + 1]
        
    return idx

if __name__ == "__main__":
    # 1. Setup
    VOCAB_SIZE = 50257
    EMBEDDING_DIM = 256
    CONTEXT_LENGTH = 64
    NUM_HEADS = 8
    NUM_LAYERS = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model (In a real scenario, you would load your saved weights here via torch.load)
    model = GPTModel(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_LENGTH, 0.0, NUM_HEADS, NUM_LAYERS)
    
    #laod the model
    model.load_state_dict(torch.load("gpt_prototype.pth", map_location=device,weights_only=True))

    model.to(device)
    model.eval() # CRITICAL: Turns off Dropout for inference
    
    # 2. The Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 3. The Prompt
    start_context = "This is a sample text"
    
    # Encode prompt to integers and shape it as a batch of 1: [1, SeqLen]
    encoded_prompt = tokenizer.encode(start_context, allowed_special={"<|endoftext|>"})
    token_tensor = torch.tensor(encoded_prompt).unsqueeze(0).to(device)
    
    print(f"Input Prompt: '{start_context}'")
    print(f"Encoded Tensor: {token_tensor.tolist()}\n")
    
    # 4. Generate
    print("Generating text...")
    out_tokens = generate_text_simple(
        model=model, 
        idx=token_tensor, 
        max_new_tokens=10, 
        context_size=CONTEXT_LENGTH
    )
    
    # 5. Decode back to English
    # out_tokens is a 2D tensor [1, final_seq_len]. We flatten it to 1D and decode.
    generated_text = tokenizer.decode(out_tokens.squeeze(0).tolist())
    
    print(f"\nFinal Output: {generated_text}")