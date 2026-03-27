import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    torch.set_float32_matmul_precision('high')

    print("Loading Hugging Face GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Compiling model... (This may take a couple of minutes on the first run)")
    model = torch.compile(model)

    input_text = "The quick brown fox jumps over the lazy dog and runs into the forest"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    tokens_to_generate = 100

    # The first forward pass triggers the actual tracing and compilation of the CUDA kernels.
    print("Warming up and triggering JIT compilation...")
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=2, use_cache=True)
        torch.cuda.synchronize()

    print("Running benchmark...")
    
    # Start Timer
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=tokens_to_generate,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Stop Timer
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    generated_tokens = output_ids.shape[1] - input_ids.shape[1]
    seconds = end_time - start_time
    tokens_per_second = generated_tokens / seconds

    print("\n--- Benchmark Results ---")
    print(f"Hardware utilized:    {device.upper()}")
    print(f"Total execution time: {seconds:.4f} seconds")
    print(f"Tokens generated:     {generated_tokens}")
    print(f"Throughput:           {tokens_per_second:.2f} tok/s")
    print(f"Avg time per token:   {(seconds * 1000.0 / generated_tokens):.2f} ms/tok")

if __name__ == "__main__":
    main()
