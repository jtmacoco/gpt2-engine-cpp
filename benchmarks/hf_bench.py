import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    print("Loading Hugging Face GPT-2 model...")
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Move to GPU if available for the hardware baseline
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.to(device)
    model.eval()

    input_text = "The quick brown fox jumps over the lazy dog and runs into the forest"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    tokens_to_generate = 100

    # Start Timer
    start_time = time.perf_counter()

    with torch.no_grad():
        # use_cache=False forces PyTorch to recalculate the whole sequence, 
        # mimicking your current C++ implementation exactly.
        output_ids = model.generate(
            input_ids,
            max_new_tokens=tokens_to_generate,
            do_sample=False, # Force greedy decoding (argmax) like your C++ code
            use_cache=False, 
            pad_token_id=tokenizer.eos_token_id
        )

    # Stop Timer
    end_time = time.perf_counter()

    # Calculate Metrics
    generated_tokens = output_ids.shape[1] - input_ids.shape[1]
    seconds = end_time - start_time
    tokens_per_second = generated_tokens / seconds

    print(f"Hardware utilized:    {device.upper()}")
    print(f"Total execution time: {seconds:.4f} seconds")
    print(f"Tokens generated:     {generated_tokens}")
    print(f"Throughput:           {tokens_per_second:.2f} tok/s")
    print(f"Avg time per token:   {(seconds * 1000.0 / generated_tokens):.2f} ms/tok")

if __name__ == "__main__":
    main()
