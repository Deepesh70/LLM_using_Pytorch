import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# The setup
# ----------
# torch is PyTorch: the mathematical engine used to build and train neural
# networks.
#
# Dataset is PyTorch's blueprint for organizing training examples. By
# inheriting from Dataset, GPTDatasetV1 promises to implement the methods
# PyTorch expects: __len__ and __getitem__.
#
# DataLoader is the batching machine that can later read this Dataset and feed
# examples into a model in mini-batches.
#
# tiktoken is OpenAI's tokenizer library. It converts raw text into integer
# token IDs, which are the actual values a GPT-style model learns from.


class GPTDatasetV1(Dataset):
    """Turn one long text string into input/target token chunks for GPT training.

    Each training example contains:
    - input_ids: a window of tokens the model is allowed to look at
    - target_ids: the same window shifted one token to the right

    That one-token shift teaches the model the core GPT task: given the current
    tokens, predict the next token at every position.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        """Prepare all training chunks when the dataset object is created.

        Args:
            txt: Raw text string to train on.
            tokenizer: Active tiktoken encoder.
            max_length: Number of tokens in each input window, also called the
                context window.
            stride: Number of tokens to move forward before taking the next
                window. Smaller strides create more overlapping examples.
        """
        self.input_ids = []
        self.target_ids = []

        # Convert the complete raw text into one long list of integer token IDs.
        # allowed_special keeps tiktoken from rejecting the document separator
        # token if the text contains "<|endoftext|>".
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        
        # Slide a fixed-size window across the token IDs.
        #
        # The loop stops at len(token_ids) - max_length so each input chunk and
        # its one-token-shifted target chunk both have exactly max_length items.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # Convert normal Python lists into PyTorch tensors, then store them
            # as one permanent training example.
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Return the total number of training examples in this dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Return the input and target tensors for one requested example."""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True
):
    """Create a PyTorch DataLoader for GPT-style next-token prediction.

    This helper hides the setup steps:
    1. Load the GPT-2 tokenizer from tiktoken.
    2. Turn the raw text into a GPTDatasetV1.
    3. Wrap that dataset in a DataLoader so PyTorch can serve mini-batches.

    Args:
        txt: Raw text string to train on.
        batch_size: Number of training examples returned in each batch.
        max_length: Number of tokens in each input and target sequence.
        stride: Number of tokens the dataset window moves between examples.
        shuffle: Whether DataLoader should randomize example order each epoch.
        drop_last: Whether to discard the final batch if it is smaller than
            batch_size. This is often useful because neural-network training is
            simpler when every batch has the same shape.
    """

    # Load the GPT-2 tokenizer. GPT-style tutorials often use this tokenizer
    # because it is widely available and maps text to the same kind of integer
    # token IDs a small GPT model can learn from.
    tokenizer = tiktoken.get_encoding("gpt2")


    # Build the custom Dataset. At this point, the text is split into many
    # input/target tensor pairs.
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Wrap the Dataset in a DataLoader. The DataLoader is responsible for
    # grouping individual examples into batches and optionally shuffling them.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

if __name__ == "__main__":  #This ensures that Only run the code inside this block if I run this specific file directly."

    raw_text = "This is a simple test text to verify that the PyTorch dataloader is functioning exactly as intended and shifting the target tensors correctly." * 100
    
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=2,
        max_length=4,
        stride=2,
        shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Input IDs:\n", inputs)
    print("Target IDs:\n", targets)
    print("Input Tensor Shape:", inputs.shape)
    print("Target Tensor Shape:", targets.shape)
    
    # Fetch the gpt2 tokenizer again so we can decode the results
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("\n--- Decoding the First Batch ---")
    # We loop through the batch size (which is 2)
    for i in range(inputs.shape[0]):
        print(f"\n[Example {i+1}]")
        
        # Convert the PyTorch tensor row back into a plain Python list of integers
        input_list = inputs[i].tolist()
        target_list = targets[i].tolist()
        print("Input Token IDs: ", input_list)
        print("Target Token IDs: ", target_list)
        # Decode the individual sub-tokens to see exactly where the words split
        input_words = [tokenizer.decode([token]) for token in input_list]
        target_words = [tokenizer.decode([token]) for token in target_list]
        
        print("Input Tokens: ", input_words)
        print("Target Tokens:", target_words)
    