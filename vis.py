import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import time

def live_visualize_attention(d_in=256, context_length=8):
    # 1. Setup the plot
    plt.ion() # Turn on interactive mode for live updates
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 2. Initialize weights and simulate input (Batch Size = 1 for visualization)
    torch.manual_seed(42) # For reproducible random tensors
    x = torch.rand(1, context_length, d_in)
    
    W_query = nn.Linear(d_in, d_in)
    W_key = nn.Linear(d_in, d_in)
    
    print("--- STARTING LIVE TENSOR VISUALIZATION ---")
    
    # --- STEP 1: Q and K ---
    queries = W_query(x)
    keys = W_key(x)
    
    ax.set_title(f"Step 1: Raw Inputs Projected to Q and K\nShape: {queries.shape}")
    ax.text(0.5, 0.5, "Q and K Matrices Computed\n(Too dense to visualize 256 dims directly)", 
            ha='center', va='center', fontsize=12)
    ax.axis('off')
    plt.pause(2.5)
    ax.clear()

    # --- STEP 2: Raw Attention Scores (Q @ K.T) ---
    attn_scores = queries @ keys.transpose(1, 2)
    # Extract the 8x8 matrix for the single batch
    scores_2d = attn_scores[0].detach().numpy()
    
    sns.heatmap(scores_2d, annot=True, fmt=".1f", cmap="coolwarm", ax=ax, cbar=False)
    ax.set_title(f"Step 2: Raw Attention Scores (Q @ K.T)\nShape: {attn_scores.shape}")
    ax.set_xlabel("Keys (Context Words)")
    ax.set_ylabel("Queries (Current Word)")
    plt.pause(3.0)
    ax.clear()

    # --- STEP 3: Apply Causal Mask ---
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    attn_scores.masked_fill_(mask.bool(), -torch.inf)
    
    scores_masked_2d = attn_scores[0].detach().numpy()
    
    sns.heatmap(scores_masked_2d, annot=True, fmt=".1f", cmap="coolwarm", ax=ax, cbar=False)
    ax.set_title(f"Step 3: Applied Causal Mask (-inf)\nUpper Triangle Blocked")
    ax.set_xlabel("Keys (Context Words)")
    ax.set_ylabel("Queries (Current Word)")
    plt.pause(3.0)
    ax.clear()

    # --- STEP 4: Softmax (The Final Weights) ---
    attn_weights = torch.softmax(attn_scores / (d_in ** 0.5), dim=-1)
    
    weights_2d = attn_weights[0].detach().numpy()
    
    sns.heatmap(weights_2d, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(f"Step 4: Softmax Applied (Attention Weights)\nRows now sum to 1.0")
    ax.set_xlabel("Keys (Context Words)")
    ax.set_ylabel("Queries (Current Word)")
    
    print("\nNotice how the upper right triangle is exactly 0.00.")
    print("This proves future tokens have zero influence on the current token.")
    
    plt.ioff() # Turn off interactive mode
    plt.show() # Keep the final plot open

if __name__ == "__main__":
    live_visualize_attention()