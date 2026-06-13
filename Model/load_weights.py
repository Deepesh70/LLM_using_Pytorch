import os
import torch
import urllib.request

# Fix SSL certificate path if environment variable points to a non-existent file
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        del os.environ["SSL_CERT_FILE"]

from transformers import GPT2LMHeadModel
from gpt import GPTModel

def map_huggingface_to_custom(custom_model, hf_model):
    print("Beginning tensor transplant...")
    
    # 1. Embeddings
    custom_model.tok_emb.weight.data = hf_model.transformer.wte.weight.data.clone()
    custom_model.pos_emb.weight.data = hf_model.transformer.wpe.weight.data.clone()

    # 2. Transformer Blocks (All 12 Layers)
    for i in range(12):
        b_custom = custom_model.trf_blocks[i]
        b_hf = hf_model.transformer.h[i]

        # A. Attention Weights
        # Hugging Face stores Q, K, V as one giant tensor. We must split it into three 768-D chunks.
        # They also use Conv1D instead of Linear, so we must mathematically transpose (.T) the matrices.
        c_attn_weights = b_hf.attn.c_attn.weight.data.T
        c_attn_bias = b_hf.attn.c_attn.bias.data

        q_w, k_w, v_w = torch.split(c_attn_weights, 768, dim=0)
        q_b, k_b, v_b = torch.split(c_attn_bias, 768, dim=0)

        b_custom.attn.W_query.weight.data = q_w.clone()
        b_custom.attn.W_query.bias.data = q_b.clone()
        b_custom.attn.W_key.weight.data = k_w.clone()
        b_custom.attn.W_key.bias.data = k_b.clone()
        b_custom.attn.W_value.weight.data = v_w.clone()
        b_custom.attn.W_value.bias.data = v_b.clone()

        # B. Attention Output Projection
        b_custom.attn.out_proj.weight.data = b_hf.attn.c_proj.weight.data.T.clone()
        b_custom.attn.out_proj.bias.data = b_hf.attn.c_proj.bias.data.clone()

        # C. Layer Normalizations
        b_custom.norm1.weight.data = b_hf.ln_1.weight.data.clone()
        b_custom.norm1.bias.data = b_hf.ln_1.bias.data.clone()
        b_custom.norm2.weight.data = b_hf.ln_2.weight.data.clone()
        b_custom.norm2.bias.data = b_hf.ln_2.bias.data.clone()

        # D. Feed Forward Network
        b_custom.ff.net[0].weight.data = b_hf.mlp.c_fc.weight.data.T.clone()
        b_custom.ff.net[0].bias.data = b_hf.mlp.c_fc.bias.data.clone()
        b_custom.ff.net[2].weight.data = b_hf.mlp.c_proj.weight.data.T.clone()
        b_custom.ff.net[2].bias.data = b_hf.mlp.c_proj.bias.data.clone()

    # 3. Final Norm and Output Head
    custom_model.final_norm.weight.data = hf_model.transformer.ln_f.weight.data.clone()
    custom_model.final_norm.bias.data = hf_model.transformer.ln_f.bias.data.clone()
    custom_model.out_head.weight.data = hf_model.lm_head.weight.data.clone()

    print("Transplant successful.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    # 1. Initialize YOUR blank architecture with true GPT-2 dimensions
    # CRITICAL: qkv_bias=True is required because OpenAI used biases in their attention matrices.
    custom_gpt = GPTModel(
        vocab_size=50257, 
        embedding_dim=768, 
        context_length=1024, 
        drop_rate=0.0, 
        num_heads=12, 
        num_layers=12,
        qkv_bias=True 
    )

    # 2. Download OpenAI's official pre-trained model
    print("Downloading GPT-2 (124M) from Hugging Face. This may take a minute...")
    hf_gpt = GPT2LMHeadModel.from_pretrained("gpt2")

    # 3. Force the weights into your architecture
    map_huggingface_to_custom(custom_gpt, hf_gpt)

    # 4. Save your new, highly intelligent model
    save_path = "gpt2_124m_custom.pth"
    torch.save(custom_gpt.state_dict(), save_path)
    print(f"Model saved locally as {save_path}. You now have a functional LLM.")