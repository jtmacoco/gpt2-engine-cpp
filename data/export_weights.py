# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import torch
import numpy

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

state_dict = model.state_dict()

embeddings = state_dict['transformer.wte.weight'].detach().numpy()

embeddings.tofile("weights_embeddings.bin")
print(f"Exported embeddings shape {embeddings.shape}")
