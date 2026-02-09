from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# 1. Setup Model
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model.eval()

# 2. Prepare Input
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
# "Hello world" -> IDs [15496, 995]

# 3. Setup a "Hook" to catch the data right after the QKV projection
# We want the output of: transformer.h[0].attn.c_attn
captured_qkv = None

def get_qkv_output(name):
    def hook(model, input, output):
        global captured_qkv
        captured_qkv = output.detach()
    return hook

# Attach the hook to the first layer's attention projection
model.transformer.h[0].attn.c_attn.register_forward_hook(get_qkv_output("qkv_proj"))

# 4. Run the model
with torch.no_grad():
    model(**inputs)

# 5. Export and Print
print(f"Captured QKV Shape: {captured_qkv.shape}") # Should be [1, 2, 2304]

# Get the data for the first token ("Hello")
token_0_data = captured_qkv[0][0].numpy()

print("\n--- GROUND TRUTH (Python) ---")
print(f"First 5 values: {token_0_data[:5]}")
print(f"Last 5 values:  {token_0_data[-5:]}")

# Optional: Save to file if you want to load it in C++ for automated testing
# token_0_data.tofile("debug_qkv_output.bin")
