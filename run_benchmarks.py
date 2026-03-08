import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
EXECUTABLE = "./build/benchmarks/gpu_bench"
TOKEN_SIZES = [1, 2, 5, 10, 20, 30, 50, 100, 200, 300, 500, 700, 800, 900, 1000]
#TOKEN_SIZES = [1, 2, 5, 10, 15, 20, 30, 50]

# Match a markdown table data row:
# | Tokens | TTFT | TPOT | Total | Throughput |
ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$",
    re.MULTILINE
)

def run_benchmark(tokens: int):
    print(f"Running benchmark for {tokens} tokens...") 
    result = subprocess.run([EXECUTABLE, str(tokens)], capture_output=True, text=True)

    # If your binary prints errors to stderr, surface them
    if result.returncode != 0:
        print(f"Benchmark failed for {tokens} tokens (exit {result.returncode}).\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}")
        return None

    output = result.stdout

    # Find all rows; take the last one (safest if banner repeats)
    matches = ROW_RE.findall(output)
    if not matches:
        print(f"Error parsing output for {tokens} tokens:\n{output}")
        return None

    row = matches[-1]
    parsed_tokens = int(row[0])
    ttft = float(row[1])
    tpot = float(row[2])
    total_s = float(row[3])
    throughput = float(row[4])

    # Sanity check: token count in row should match requested tokens
    if parsed_tokens != tokens:
        print(f"Warning: parsed token count ({parsed_tokens}) != requested ({tokens}). Using parsed value.")

    return {
        "Tokens": parsed_tokens,
        "TTFT (ms)": ttft,
        "TPOT (ms/tok)": tpot,
        "Total Time (s)": total_s,
        "Throughput (tok/s)": throughput
    }

def main():
    results = []
    for t in TOKEN_SIZES:
        res = run_benchmark(t)
        if res:
            results.append(res)

    if not results:
        print("No results collected. Exiting.")
        return

    df = pd.DataFrame(results).sort_values("Tokens").reset_index(drop=True)

    print("\n" + "=" * 50)
    print(" BENCHMARK RESULTS TABLE")
    print("=" * 50)
    print(df.to_markdown(index=False))

    # --- Graphs (kept similar to your original) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #fig.suptitle("cu-gpt-engine Performance Metrics (KV Cache Enabled)", fontsize=16)
    fig.suptitle("cu-gpt-engine Performance Metrics (KV Cache)", fontsize=16)

    axes[0, 0].plot(df["Tokens"], df["TTFT (ms)"], marker='o')
    axes[0, 0].set_title("Time To First Token (TTFT)")
    axes[0, 0].set_xlabel("Tokens Generated")
    axes[0, 0].set_ylabel("Milliseconds")
    axes[0, 0].grid(True)

    axes[0, 1].plot(df["Tokens"], df["TPOT (ms/tok)"], marker='o')
    axes[0, 1].set_title("Time Per Output Token (TPOT)")
    axes[0, 1].set_xlabel("Tokens Generated")
    axes[0, 1].set_ylabel("Milliseconds / Token")
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim(0, df["TPOT (ms/tok)"].max() * 1.2)

    axes[1, 0].plot(df["Tokens"], df["Total Time (s)"], marker='o')
    axes[1, 0].set_title("Total Execution Time")
    axes[1, 0].set_xlabel("Tokens Generated")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].grid(True)

    axes[1, 1].plot(df["Tokens"], df["Throughput (tok/s)"], marker='o')
    axes[1, 1].set_title("Overall Throughput")
    axes[1, 1].set_xlabel("Tokens Generated")
    axes[1, 1].set_ylabel("Tokens / Second")
    axes[1, 1].grid(True)

    plt.tight_layout()

    out_dir = Path("./benchmarks/gpu_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results_gpu.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nGraphs saved to '{out_path}'")

if __name__ == "__main__":
    main()
