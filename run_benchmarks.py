import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
EXECUTABLE = "./build/benchmarks/gpu_bench" 
TOKEN_SIZES = [20, 50, 100, 200, 300, 500, 700, 800, 900, 1000]

def run_benchmark(tokens):
    print(f"Running benchmark for {tokens} tokens...")
    result = subprocess.run([EXECUTABLE, str(tokens)], capture_output=True, text=True)
    output = result.stdout

    #Regex patterns to extract metrics from your C++ stdout
    ttft_match = re.search(r"TTFT \(Prefill\):\s+([\d.]+)", output)
    tpot_match = re.search(r"TPOT \(Decode Avg\):\s+([\d.]+)", output)
    time_match = re.search(r"Total execution time:\s+([\d.]+)", output)
    throughput_match = re.search(r"Overall Throughput:\s+([\d.]+)", output)

    if not all([ttft_match, tpot_match, time_match, throughput_match]):
        print(f"Error parsing output for {tokens} tokens:\n{output}")
        return None

    return {
        "Tokens": tokens,
        "TTFT (ms)": float(ttft_match.group(1)),
        "TPOT (ms/tok)": float(tpot_match.group(1)),
        "Total Time (s)": float(time_match.group(1)),
        "Throughput (tok/s)": float(throughput_match.group(1))
    }

def main():
    results = []
    for tokens in TOKEN_SIZES:
        res = run_benchmark(tokens)
        if res:
            results.append(res)
    
    #Create and Display Table
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(" BENCHMARK RESULTS TABLE")
    print("="*50)
    print(df.to_markdown(index=False))
    
    #Generate Graphs
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("cu-gpt-engine Performance Metrics (KV Cache Enabled)", fontsize=16)

    # Top-Left: TTFT (Should be mostly flat since prompt is same size)
    axes[0, 0].plot(df["Tokens"], df["TTFT (ms)"], marker='o', color='purple')
    axes[0, 0].set_title("Time To First Token (TTFT)")
    axes[0, 0].set_xlabel("Tokens Generated")
    axes[0, 0].set_ylabel("Milliseconds")
    axes[0, 0].grid(True)

    #Top-Right: TPOT (The true test of O(n) KV Cache scaling)
    axes[0, 1].plot(df["Tokens"], df["TPOT (ms/tok)"], marker='o', color='blue')
    axes[0, 1].set_title("Time Per Output Token (TPOT)")
    axes[0, 1].set_xlabel("Tokens Generated")
    axes[0, 1].set_ylabel("Milliseconds / Token")
    axes[0, 1].grid(True)
    # Set Y-axis to start at 0 to show how flat the curve is
    axes[0, 1].set_ylim(0, df["TPOT (ms/tok)"].max() * 1.2) 

    # Bottom-Left: Total Time
    axes[1, 0].plot(df["Tokens"], df["Total Time (s)"], marker='o', color='red')
    axes[1, 0].set_title("Total Execution Time")
    axes[1, 0].set_xlabel("Tokens Generated")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].grid(True)

    # Bottom-Right: Throughput
    axes[1, 1].plot(df["Tokens"], df["Throughput (tok/s)"], marker='o', color='green')
    axes[1, 1].set_title("Overall Throughput")
    axes[1, 1].set_xlabel("Tokens Generated")
    axes[1, 1].set_ylabel("Tokens / Second")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=300)
    print("\nGraphs saved to 'benchmark_results.png'")

if __name__ == "__main__":
    main()
