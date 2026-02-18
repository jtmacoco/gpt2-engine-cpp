# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import torch
import numpy

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

state_dict = model.state_dict()

wte = state_dict['transformer.wte.weight'].detach().numpy() #token embeddings
wpe = state_dict['transformer.wpe.weight'].detach().numpy() #position embedding

#Extract layer norm 1
#gamma = scale (weight), beta = shift (bias)
ln_1_g = state_dict['transformer.h.0.ln_1.weight'].detach().numpy()
ln_1_b = state_dict['transformer.h.0.ln_1.bias'].detach().numpy()

layer_idx = 0

#QKV Projections (Fused)
qkv_w = state_dict[f'transformer.h.{layer_idx}.attn.c_attn.weight'].detach().numpy()
qkv_b = state_dict[f'transformer.h.{layer_idx}.attn.c_attn.bias'].detach().numpy()

#Attention Output Projection
attn_proj_w = state_dict[f'transformer.h.{layer_idx}.attn.c_proj.weight'].detach().numpy()
attn_proj_b = state_dict[f'transformer.h.{layer_idx}.attn.c_proj.bias'].detach().numpy()

# Extract Layer Norm 2
ln_2_g = state_dict[f'transformer.h.{layer_idx}.ln_2.weight'].detach().numpy()
ln_2_b = state_dict[f'transformer.h.{layer_idx}.ln_2.bias'].detach().numpy()

# 2. Extract MLP / Feed-Forward weights
# c_fc is the "up-scaling" layer (768 -> 3072)
fc_w = state_dict[f'transformer.h.{layer_idx}.mlp.c_fc.weight'].detach().numpy()
fc_b = state_dict[f'transformer.h.{layer_idx}.mlp.c_fc.bias'].detach().numpy()

# c_proj is the "down-scaling" layer (3072 -> 768)
proj_w = state_dict[f'transformer.h.{layer_idx}.mlp.c_proj.weight'].detach().numpy()
proj_b = state_dict[f'transformer.h.{layer_idx}.mlp.c_proj.bias'].detach().numpy()

with open("gpt2_embeddings.bin", "wb") as f:
    wte.tofile(f) # Write WTE first
    wpe.tofile(f) # Write WPE immediately after
    ln_1_g.tofile(f)
    ln_1_b.tofile(f)

    #Attention weights
    qkv_w.tofile(f)
    qkv_b.tofile(f)
    attn_proj_w.tofile(f)
    attn_proj_b.tofile(f)

    ln_2_g.tofile(f)
    ln_2_b.tofile(f)
    fc_w.tofile(f)
    fc_b.tofile(f)
    proj_w.tofile(f)
    proj_b.tofile(f)

#print(f"Exported WTE {wte.shape} and WPE {wpe.shape}")
with open("export_weights_info.txt", "w") as f:
    print("Exported:", file=f)
    print(f" - WTE:                    {wte.shape}", file=f)
    print(f" - WPE:                    {wpe.shape}", file=f)
    print(f" - LN1 Gamma:              {ln_1_g.shape}", file=f)
    print(f" - LN1 Beta:               {ln_1_b.shape}", file=f)
    print(f" - QKV Weights:            {qkv_w.shape}", file=f)
    print(f" - QKV Bias:               {qkv_b.shape}", file=f)
    print(f" - Proj Attention Weights: {attn_proj_w.shape}", file=f)
    print(f" - Proj Attention Bias:    {attn_proj_b.shape}", file=f)
    print(f" - LN2 GAMMA:              {ln_2_g.shape}", file=f)
    print(f" - LN2 Beta:               {ln_2_b.shape}", file=f)
    print(f" - FC Weights:             {fc_w.shape}", file=f)
    print(f" - FC Bias:                {fc_b.shape}", file=f)
    print(f" - Proj Weights:           {proj_w.shape}", file=f)
    print(f" - Proj Bias:              {proj_b.shape}", file=f)




# Check specific indices to see if there's actual data
print(f"QKV Start: {qkv_w.flatten()[:5]}")
print(f"QKV Mid:   {qkv_w.flatten()[1000:1005]}")
