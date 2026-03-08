import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Model

def load_tensor(filename, seq_len, hidden_dim):
    data = np.loadtxt(filename, dtype=np.float32)
    return torch.tensor(data).view(1, seq_len, hidden_dim)

def main():
    seq_len = 4
    hidden_dim = 768 

    print("Loading Hugging Face GPT-2 model...")
    model = GPT2Model.from_pretrained("gpt2").eval()

    print("Loading C++ tensors...")
    h_input = load_tensor("h_input.txt", seq_len, hidden_dim)
    cuda_out = load_tensor("h_output.txt", seq_len, hidden_dim)

    with torch.no_grad():
        x = model.h[0].ln_1(h_input)
        hf_out = model.h[0].attn(x)[0]

    mse = F.mse_loss(cuda_out, hf_out).item()
    max_diff = torch.max(torch.abs(cuda_out - hf_out)).item()

    print("-" * 30)
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Max Absolute Difference: {max_diff:.8f}")

    tolerance = 1e-3 
    if max_diff < tolerance:
        print(" TEST PASSED: C++ Attention Layer matches Hugging Face")
    else:
        print(" TEST FAILED: Differences exceed tolerance.")

if __name__ == "__main__":
    main()
