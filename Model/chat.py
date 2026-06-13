import torch
import tiktoken
from gpt import GPTModel

def format_prompt(instruction):
    # This MUST match your training format perfectly
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

def generate_chat(model, tokenizer, prompt, max_new_tokens, context_size, temperature=0.8, top_k=40):
    # 1. Format the user's prompt into the SFT template
    formatted_prompt = format_prompt(prompt)
    idx = torch.tensor(tokenizer.encode(formatted_prompt)).unsqueeze(0).to(model.out_head.weight.device)
    
    stop_token = 50256 # <|endoftext|>

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] # Focus on the last step
        

        # --- NEW: Repetition Penalty ---
        # Mathematically crush the probability of tokens we have already used.
        repetition_penalty = 1.2
        for token_id in set(idx[0].tolist()):
            score = logits[0, token_id]
            if score < 0:
                logits[0, token_id] = score * repetition_penalty
            else:
                logits[0, token_id] = score / repetition_penalty

                

        # 2. Temperature Scaling
        # Divides the raw logits by the temperature. 
        # T < 1.0 makes the model more confident (sharper distribution).
        # T > 1.0 makes the model more creative (flatter distribution).
        if temperature > 0.0:
            logits = logits / temperature
            
        # 3. Top-K Filtering
        # Chops off the long tail of low-probability garbage tokens
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 4. Convert Logits to Probabilities and Sample
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 5. Early Stop Trigger
        # If the model predicts 50256, it has finished its answer. We break the loop immediately.
        if idx_next.item() == stop_token:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
        
    # Decode only the generated response, cutting off the prompt template
    response_tokens = idx[0].tolist()[len(tokenizer.encode(formatted_prompt)):]
    return tokenizer.decode(response_tokens)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing Inference Engine...")
    
    # 1. Initialize the 124M Architecture
    model = GPTModel(
        vocab_size=50257, embedding_dim=768, context_length=1024, 
        drop_rate=0.0, num_heads=12, num_layers=12, qkv_bias=True
    )
    
    # 2. Load Your SFT Weights
    model.load_state_dict(torch.load("gpt2_chatbot_sft.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Disable dropout
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("\n--- Chatbot Ready ---")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        response = generate_chat(
            model=model, 
            tokenizer=tokenizer, 
            prompt=user_input, 
            max_new_tokens=150, 
            context_size=1024,
            temperature=0.7, # 0.7 is a standard balance of logic and creativity
            top_k=40
        )
        print(f"Bot: {response}\n")