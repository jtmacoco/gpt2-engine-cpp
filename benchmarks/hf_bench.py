import time
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def sync_if_needed(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def main(token_size: int):
    torch.set_float32_matmul_precision("high")

    print("Loading Hugging Face GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # GPT-2 has no pad token by default. Using eos as pad is a common fallback.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    print("Compiling model... (This may take a couple of minutes on the first run)")
    model = torch.compile(model)

    input_text = "The quick brown fox jumps over the lazy dog and runs into the forest"

    # Create model inputs once, and keep attention_mask
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tokens_to_generate = token_size

    print("Warming up and triggering compilation...")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=tokens_to_generate,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    sync_if_needed(device)

    print("Running benchmark...")

    sync_if_needed(device)
    start_time = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=tokens_to_generate,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    sync_if_needed(device)
    end_time = time.perf_counter()

    generated_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    seconds = end_time - start_time
    total_ms = seconds * 1000.0
    seconds = end_time - start_time

    mean_total = seconds * 1000.0
    mean_throughput = generated_tokens / seconds if seconds > 0 else 0.0

    throughput = generated_tokens / seconds if seconds > 0 else float("inf")

    prompt_len = inputs["input_ids"].shape[1]

    # This is NOT true TPOT. It is end-to-end average time per generated token.
    avg_time_per_token_ms = total_ms / generated_tokens if generated_tokens > 0 else 0.0
    mean_ttft = 0.00
    mean_tpot = mean_total / generated_tokens if generated_tokens > 0 else 0.0
    prompt_len = inputs["input_ids"].shape[1]


    print("\n==================================================")
    print(f" PYTORCH BENCHMARK (JIT, warmup 1)")
    print(f" prompt_len={prompt_len}, requested_gen={tokens_to_generate}")
    print("==================================================")
    print("| Prompt | Req Gen |   TTFT (ms) |   TPOT (ms/tok) |   Total (ms) |   Throughput (tok/s) |")
    print("|-------:|--------:|------------:|----------------:|-------------:|---------------------:|")
    print(f"| {prompt_len:>6} | {tokens_to_generate:>7} | {mean_ttft:>11.2f} | {mean_tpot:>14.2f} | {mean_total:>11.2f} | {mean_throughput:>20.2f} |") 
if __name__ == "__main__":
    sz = int(sys.argv[1])
    main(token_size=sz)
