import torch
from transformers import GPT2Model, GPT2Tokenizer

# 1. Setup Model
model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 2. Prepare Input
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
print(f"Token IDs: {inputs.input_ids}")

# 3. Extract Layer 0 Components
block = model.h[0]
ln_1 = block.ln_1
attn = block.attn

# 4. Run Manual Forward Pass (Input -> LN -> Attn)
with torch.no_grad():
    # A. Get Embeddings (Input)
    wte = model.wte(inputs.input_ids)
    wpe = model.wpe(torch.arange(inputs.input_ids.size(1)).unsqueeze(0))
    x = wte + wpe # This is your 'input_buffer'
    
    # B. Layer Norm
    x_ln = ln_1(x) # This is your 'ln_buffer'
    
    # C. Attention
    # attn returns (output, present_key_values)
    attn_output, _ = attn(x_ln) # This is your 'attention_output'

    # 5. Print First 10 Values
    print("\n--- Python Reference Output (First 10 values) ---")
    # Get first token, first 10 dimensions
    print(attn_output[0, 0, :10].numpy())
