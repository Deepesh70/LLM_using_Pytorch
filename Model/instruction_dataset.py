import json
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        print(f"Loading SFT data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        self.tokenizer = tokenizer
        self.max_length = max_length

        # GPT-2 does not have a dedicated padding token.
        # We reuse the End-Of-Text token (50256) to fill empty space.
        self.pad_token_id = 50256 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        
        # 1. Inject the data into the structural mold
        prompt = self.format_alpaca(item)
        
        # 2. Convert to integers
        encoded = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        
        # 3. Truncate: If the prompt is too long, we chop off the excess.
        # We add +1 because we need to shift the targets later.
        if len(encoded) > self.max_length + 1:
            encoded = encoded[:self.max_length + 1]
            
        # 4. Pad: If the prompt is too short, we append 50256 until it fits perfectly.
        pad_len = (self.max_length + 1) - len(encoded)
        encoded = encoded + [self.pad_token_id] * pad_len
        
        # 5. Shift for Autoregressive Learning
        # x is what the model sees. y is what the model must predict.
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        
        return x, y

    def format_alpaca(self, item):
        # This exact string structure is what the matrix weights will memorize.
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        if input_text:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}<|endoftext|>"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output_text}<|endoftext|>"

def create_sft_dataloader(data_path, batch_size, max_length):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructionDataset(data_path, tokenizer, max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True # Drops the final batch if it doesn't divide evenly into the batch size
    )
    return dataloader




# dummy data
if __name__ == "__main__":
    # Test parameters
    BATCH_SIZE = 2
    MAX_LENGTH = 128
    
    dataloader = create_sft_dataloader("dummy/dummy_alpaca.json", BATCH_SIZE, MAX_LENGTH)
    
    # Grab one batch
    for x, y in dataloader:
        print(f"Input Shape (X): {x.shape}")
        print(f"Target Shape (Y): {y.shape}")
        
        # Verify padding physically exists
        print(f"\nLast 10 tokens of X[0]: {x[0][-10:].tolist()}")
        break