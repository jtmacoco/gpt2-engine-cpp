# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import torch
import numpy

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

state_dict = model.state_dict()

wte = state_dict['transformer.wte.weight'].detach().numpy() #token embeddings
wpe = state_dict['transformer.wpe.weight'].detach().numpy() #position embedding

#extract layer norm 1
#gamma = scale (weight), beta = shift (bias)
ln_1_g = state_dict['transformer.h.0.ln_1.weight'].detach().numpy()
ln_1_b = state_dict['transformer.h.0.ln_1.bias'].detach().numpy()

with open("gpt2_embeddings.bin", "wb") as f:
    wte.tofile(f) # Write WTE first
    wpe.tofile(f) # Write WPE immediately after
    ln_1_g.tofile(f)
    ln_1_b.tofile(f)

#print(f"Exported WTE {wte.shape} and WPE {wpe.shape}")
print(f"Exported:")
print(f" - WTE: {wte.shape}")
print(f" - WPE: {wpe.shape}")
print(f" - LN1 Gamma: {ln_1_g.shape}")
print(f" - LN1 Beta:  {ln_1_b.shape}")
