"""Prepare tokenized training batches for the GPT model.

Data flow:
1. Start with one long raw text string.
2. Tokenize it with the GPT-2 tokenizer from tiktoken.
3. Slice the token list into fixed-length input windows.
4. Create target windows shifted one token to the right.
5. Wrap those examples in a PyTorch DataLoader for batching.

Example:
input tokens:  [This, is, a, sample]
target tokens: [is,   a,  sample, text]

That shift teaches the model to predict the next token.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset

# Fix SSL certificate path if environment variable points to a non-existent file
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        del os.environ["SSL_CERT_FILE"]

import tiktoken


class GPTDatasetV1(Dataset):
    """Turn one long token stream into next-token prediction examples."""

    def __init__(self, txt, tokenizer, max_length, stride):
        """Create all training examples.

        Args:
            txt: Raw text used for training.
            tokenizer: GPT-2 tokenizer used to convert text to token IDs.
            max_length: Number of input tokens per example.
            stride: Number of tokens to move forward for the next example.
                Smaller stride means more overlapping examples.
        """

        self.input_ids = []
        self.target_ids = []

        # Convert complete text into integer token IDs. These IDs are what the
        # embedding layer receives during training.
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Build fixed-length input and target chunks. The target starts one
        # token later than the input, so every input position has a label for
        # "what token should come next?"
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        """Return how many input/target examples were created."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Return one input sequence and its shifted target sequence."""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
):
    """Create a DataLoader for GPT next-token training.

    Returns batches shaped like:
    input_batch:  [batch_size, max_length]
    target_batch: [batch_size, max_length]
    """

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


if __name__ == "__main__":
    # Quick standalone test showing that targets are shifted by one token.
    raw_text = (
        "This is a simple test text to verify that the PyTorch dataloader is "
        "functioning exactly as intended and shifting the target tensors "
        "correctly. "
    ) * 100

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=2,
        max_length=4,
        stride=2,
        shuffle=False,
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Input IDs:\n", inputs)
    print("Target IDs:\n", targets)
    print("Input Tensor Shape:", inputs.shape)
    print("Target Tensor Shape:", targets.shape)

    tokenizer = tiktoken.get_encoding("gpt2")

    print("\n--- Decoding the First Batch ---")
    for i in range(inputs.shape[0]):
        print(f"\n[Example {i + 1}]")

        input_list = inputs[i].tolist()
        target_list = targets[i].tolist()

        print("Input Token IDs: ", input_list)
        print("Target Token IDs: ", target_list)

        input_words = [tokenizer.decode([token]) for token in input_list]
        target_words = [tokenizer.decode([token]) for token in target_list]

        print("Input Tokens: ", input_words)
        print("Target Tokens:", target_words)
