import torch
import torch.nn as nn
from gpt import GPTModel
from instruction_dataset import create_sft_dataloader

def main():
    # 1. HYPERPARAMETERS FOR SFT
    VOCAB_SIZE = 50257
    EMBEDDING_DIM = 768
    CONTEXT_LENGTH = 1024
    NUM_HEADS = 12
    NUM_LAYERS = 12
    
    BATCH_SIZE = 4        # Keep it low to prevent VRAM OOM on local GPUs
    EPOCHS = 1              # increase afterwards after testing
    LEARNING_RATE = 5e-5  # Aggressively low to protect pre-trained knowledge

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing SFT on device: {device}")

    # 2. DATA LOADERS
    # For a real run, replace these paths with your true alpaca_data.json split
    train_loader = create_sft_dataloader("alpaca_data.json", batch_size=BATCH_SIZE, max_length=CONTEXT_LENGTH)

    # 3. INITIALIZE ARCHITECTURE & TRANSPLANT WEIGHTS
    model = GPTModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_length=CONTEXT_LENGTH,
        drop_rate=0.0,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        qkv_bias=True
    )
    
    # Load the custom GPT-2 checkpoint we generated in the previous stage
    print("Loading pre-trained GPT-2 checkpoint...")
    model.load_state_dict(torch.load("gpt2_124m_custom.pth", map_location=device, weights_only=True))
    model.to(device)

    # 4. OPTIMIZER & MASKED LOSS CRITERION
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # CRITICAL: ignore_index=50256 tells PyTorch to completely ignore the pad tokens 
    # during backpropagation so the model doesn't optimize for predicting empty space.
    criterion = nn.CrossEntropyLoss(ignore_index=50256)

    # 4.5 INITIALIZE THE AMP SCALER (Add this right before the epoch loop)
    scaler = torch.amp.GradScaler('cuda')

    print("Beginning Supervised Fine-Tuning with Mixed Precision...")
    
    # 5. SFT TRAINING LOOP
    print("Beginning Supervised Fine-Tuning with Mixed Precision...")
    
    # Wrap the entire training process in a try block to catch manual stops
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            
            for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                optimizer.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):       #change to bfloat if supports
                    logits = model(input_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                
                # Print status
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
                # OPTIONAL: Save a regular checkpoint every 1000 batches just in case of a crash
                if batch_idx % 1000 == 0 and batch_idx > 0:
                    torch.save(model.state_dict(), f"gpt2_checkpoint_batch_{batch_idx}.pth")
                    print(f"\n[Auto-Save] Periodic checkpoint saved at batch {batch_idx}")

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Average SFT Loss: {avg_loss:.4f}")

        # Regular successful save
        torch.save(model.state_dict(), "gpt2_chatbot_sft.pth")
        print("SFT complete successfully. Final chatbot weights saved.")

    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("TRAINING INTERRUPTED BY USER (Ctrl+C detected).")
        print("Executing emergency save sequence...")
        
        # Save whatever progress the model has made up to this exact batch
        emergency_path = "gpt2_chatbot_interrupted.pth"
        torch.save(model.state_dict(), emergency_path)
        
        print(f"Emergency checkpoint saved to: {emergency_path}")
        print("You can safely close the terminal now without losing progress.")
        print("="*50)


if __name__ == "__main__":
    main()