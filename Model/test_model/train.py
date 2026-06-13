"""Train the small GPT model on a toy text corpus.

Full training flow:
1. Prepare raw text and split it into input/target token windows.
2. Build GPTModel with token embeddings, positional embeddings, transformer
   blocks, final normalization, and a vocabulary projection head.
3. For each batch, predict the next token at every position.
4. Compare predictions with shifted target tokens using cross-entropy loss.
5. Backpropagate the loss and update model parameters.
6. Save the trained weights so generate.py can load them later.
"""

import torch
import torch.nn as nn

from dataset_prep import create_dataloader_v1
from gpt import GPTModel


MODEL_PATH = "gpt_prototype.pth"


def main():
    # Hyperparameters. Keep these matched with generate.py when loading the
    # checkpoint, because the checkpoint tensor shapes depend on them.
    VOCAB_SIZE = 50257
    EMBEDDING_DIM = 256
    CONTEXT_LENGTH = 64
    NUM_HEADS = 8
    NUM_LAYERS = 4
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # This is a tiny toy corpus. The model can only learn patterns present in
    # this repeated sentence; it will not become a general-purpose chatbot.
    training_sentence = (
        "This is a sample text for training the GPT model. "
        "It will be used to create input and target sequences for the model "
        "to learn from. "
    )
    raw_text = training_sentence * 200

    # The dataloader returns:
    # input_batch:  [batch_size, context_length]
    # target_batch: [batch_size, context_length]
    # target_batch is the same text shifted one token ahead.
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=BATCH_SIZE,
        max_length=CONTEXT_LENGTH,
        stride=CONTEXT_LENGTH // 2,
        shuffle=True,
    )

    # Build model and optimizer. Dropout is enabled during training because
    # model.train() below activates it.
    model = GPTModel(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_LENGTH, 0.1, NUM_HEADS, NUM_LAYERS)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")

    for epoch in range(EPOCHS):
        # train() enables dropout and marks the module as being in training mode.
        model.train()
        total_loss = 0.0

        for input_batch, target_batch in dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Always clear old gradients before computing the next batch loss.
            optimizer.zero_grad()

            # logits shape: [batch_size, context_length, vocab_size]
            # Each position predicts the token that should come next.
            logits = model(input_batch)

            # CrossEntropyLoss expects:
            # predictions: [number_of_items, vocab_size]
            # targets:     [number_of_items]
            # So we flatten batch and time dimensions together.
            loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))

            # Compute gradients with respect to every trainable parameter.
            loss.backward()

            # Apply one optimizer step using the gradients from this batch.
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        # Save after each epoch so generate.py can use the latest trained model.
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved after epoch {epoch + 1} to {MODEL_PATH}")


if __name__ == "__main__":
    main()
