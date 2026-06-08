import torch
import torch.nn as nn
from gpt import GPTModel
from dataset_prep import create_dataloader_v1

def main():

    # HYPERPARAMETER
    VOCAB_SIZE = 50257
    EMBEDDING_DIM = 256
    CONTEXT_LENGTH = 64
    NUM_HEADS =8 
    NUM_LAYERS = 4
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 5e-4

    device  =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # LOAD DATA
    raw_text = "This is a sample text for training the GPT model. It will be used to create input and target sequences for the model to learn from." * 200
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size = BATCH_SIZE,
        max_length = CONTEXT_LENGTH,
        stride = CONTEXT_LENGTH // 2,
        shuffle = True
    )

    # Initialize Model & Optimizer
    model = GPTModel(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_LENGTH, 0.1, NUM_HEADS, NUM_LAYERS)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
            input_batch = input_batch.to(device)
            target_batch =  target_batch.to(device)

            optimizer.zero_grad()

            logits = model(input_batch)

            loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))

            loss.backward()

            optimizer.step()

            total_loss = total_loss + loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "gpt_prototype.pth")
        print("Model saved after epoch", epoch+1)
    
if __name__ == "__main__":
    main()
    
    