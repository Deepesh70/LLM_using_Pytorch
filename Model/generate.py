"""Generate text from the small GPT model trained by train.py.

Full inference flow:
1. Build the same GPTModel architecture that was used during training.
2. Load the saved parameters from gpt_prototype.pth.
3. Convert the user's prompt into GPT-2 token IDs.
4. Feed the current token sequence into the model.
5. Take the model's prediction for the last position only.
6. Append the selected next token and repeat.
7. Decode all token IDs back into readable text.
"""

import os
import torch

# Fix SSL certificate path if environment variable points to a non-existent file
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        del os.environ["SSL_CERT_FILE"]

import tiktoken

from gpt import GPTModel


MODEL_PATH = "gpt2_124m_custom.pth"


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Autoregressively generate new tokens using greedy decoding.

    Args:
        model: Trained GPTModel in eval mode.
        idx: Prompt token IDs with shape [batch_size, current_sequence_length].
        max_new_tokens: Number of new tokens to append after the prompt.
        context_size: Maximum number of tokens the model can see at once.

    Returns:
        A token tensor containing the original prompt plus generated tokens.
    """

    # GPT generation is autoregressive: one new token is predicted, appended,
    # then used as part of the input for the next prediction.
    for _ in range(max_new_tokens):
        # Keep only the newest tokens if the sequence grows longer than the
        # model's context window. The positional embedding table has entries
        # only up to context_size, so longer inputs would fail.
        idx_cond = idx[:, -context_size:]

        # Inference does not need gradients. This saves memory and makes
        # generation faster because PyTorch does not build a backward graph.
        with torch.no_grad():
            # logits shape: [batch_size, sequence_length, vocab_size]
            logits = model(idx_cond)

        # The model predicts a next-token distribution for every input position.
        # For generation we only need the distribution after the final token.
        logits = logits[:, -1, :]

        # Greedy decoding chooses the single highest-scoring token. This is
        # deterministic but can sound repetitive; sampling can be added later.
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append the predicted token to the running sequence.
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


if __name__ == "__main__":
    # These hyperparameters must match train.py. If they differ, the saved
    # checkpoint shapes will not fit this model architecture.
    VOCAB_SIZE = 50257
    EMBEDDING_DIM = 768
    CONTEXT_LENGTH = 1024
    NUM_HEADS = 12
    NUM_LAYERS = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model architecture first, then load the trained weights.
    # Without loading weights, the model is random and will produce nonsense.
    model = GPTModel(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_LENGTH, 0.0, NUM_HEADS, NUM_LAYERS)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Run train.py first so the model has "
            "trained weights to generate from."
        )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)

    # eval() disables dropout and other training-only behavior.
    model.eval()

    # The same tokenizer must be used during training and generation so token
    # IDs mean the same thing in both scripts.
    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "The future of artificial intelligence is"

    # Convert text -> token IDs -> tensor batch of size 1.
    encoded_prompt = tokenizer.encode(start_context, allowed_special={"<|endoftext|>"})
    token_tensor = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0).to(device)

    print(f"Input Prompt: '{start_context}'")
    print(f"Encoded Tensor: {token_tensor.tolist()}\n")

    print("Generating text...")
    out_tokens = generate_text_simple(
        model=model,
        idx=token_tensor,
        max_new_tokens=10,
        context_size=CONTEXT_LENGTH,
    )

    # Convert token IDs back to text.
    generated_text = tokenizer.decode(out_tokens.squeeze(0).tolist())

    print(f"\nFinal Output: {generated_text}")
