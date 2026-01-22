# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import torch
import numpy

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

state_dict = model.state_dict()

wte = state_dict['transformer.wte.weight'].detach().numpy() #

wpe = state_dict['transformer.wpe.weight'].detach().numpy()

with open("gpt2_embeddings.bin", "wb") as f:
    wte.tofile(f) # Write WTE first
    wpe.tofile(f) # Write WPE immediately after

print(f"Exported WTE {wte.shape} and WPE {wpe.shape}")
