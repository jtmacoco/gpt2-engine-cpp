import torch
from transformers import GPT2Model, GPT2Tokenizer

# 1. Setup
model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

# 2. Run Forward Pass
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    # A. Embeddings (Token + Position)
    wte = model.wte(inputs.input_ids)
    wpe = model.wpe(torch.arange(inputs.input_ids.size(1)).unsqueeze(0))
    x = wte + wpe
    
    print("\n--- GROUND TRUTH: Embeddings (First 5) ---")
    print(x[0, 0, :5].numpy())

    # B. LayerNorm
    x_ln = model.h[0].ln_1(x)
    print("\n--- GROUND TRUTH: LN Output (First 5) ---")
    print(x_ln[0, 0, :5].numpy())
    
    # C. Attention
    attn_out, _ = model.h[0].attn(x_ln)
    print("\n--- GROUND TRUTH: Attention Output (First 10) ---")
    print(attn_out[0, 0, :10].numpy())
