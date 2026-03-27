import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
EXECUTABLE = "./build/benchmarks/cpu_bench"
RUN_CMD = ["python", "./benchmarks/hf_bench.py"] #
TOKEN_SIZES = [5, 10, 20, 30, 50, 100, 200, 300, 500, 700, 800, 900, 1000]
#TOKEN_SIZES = [5, 10, 20, 30, 50]

def _strip_cell(s: str) -> str:
    return s.strip()

def _parse_float_maybe(s: str) -> Optional[float]:
    s = s.strip()
    if not s or s.lower() in {"n/a", "na", "-"}:
        return None
    # remove any trailing units accidentally printed inside cell
    s = re.sub(r"[^\d.+-eE]", "", s)
    try:
        return float(s)
    except ValueError:
        return None

def _split_markdown_row(line: str) -> List[str]:
    # Line looks like: | col1 | col2 | ... |
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    return parts

def extract_last_markdown_table(stdout: str) -> Optional[Tuple[List[str], List[str]]]:
    """
    Returns (header_cells, last_data_row_cells) from the last markdown table in stdout.
    We look for:
        header line starting with '|'
        separator line like '|----:|---|'
        at least one data row after
    """
    lines = stdout.splitlines()
    # Identify all separator lines
    sep_idxs = []
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and re.match(r"^\|\s*:?-{3,}", line.strip()):
            sep_idxs.append(i)

    if not sep_idxs:
        return None

    # Use the LAST separator as the last table
    sep_i = sep_idxs[-1]
    if sep_i - 1 < 0:
        return None

    header_line = lines[sep_i - 1].strip()
    if not header_line.startswith("|"):
        return None

    header = _split_markdown_row(header_line)

    # Collect subsequent data rows until table ends
    data_rows = []
    for j in range(sep_i + 1, len(lines)):
        row_line = lines[j].strip()
        if not row_line.startswith("|"):
            break
        # skip if it's another separator
        if re.match(r"^\|\s*:?-{3,}", row_line):
            continue
        cells = _split_markdown_row(row_line)
        # basic sanity: row should have same number of cells as header (or close)
        if len(cells) >= 3:
            data_rows.append(cells)

    if not data_rows:
        return None

    return header, data_rows[-1]

def normalize_metrics(header: List[str], row: List[str], requested_tokens: int) -> Optional[Dict]:
    """
    Convert arbitrary table formats into a consistent metrics dict.
    Supports old format and your updated format.
    """
    # Map header->cell
    h2v = {header[i].strip(): row[i].strip() if i < len(row) else "" for i in range(len(header))}

    # Determine "Tokens" (x-axis)
    # Prefer Req Gen if present, else Tokens, else requested_tokens
    tokens_val = None
    for key in ["Tokens", "Req Gen", "Req", "Requested", "Tokens Generated"]:
        if key in h2v:
            maybe = _parse_float_maybe(h2v[key])
            if maybe is not None:
                tokens_val = int(round(maybe))
                break
    if tokens_val is None:
        tokens_val = requested_tokens

    # TTFT
    ttft = None
    for key in ["TTFT (ms)", "TTFT", "TTFT_ms"]:
        if key in h2v:
            ttft = _parse_float_maybe(h2v[key])
            break

    # TPOT
    tpot = None
    for key in ["TPOT (ms/tok)", "TPOT", "TPOT_ms_per_tok"]:
        if key in h2v:
            tpot = _parse_float_maybe(h2v[key])
            break

    # Total time: may be Total Time (s) or Total (ms) or Total Time (ms)
    total_s = None
    if "Total Time (s)" in h2v:
        total_s = _parse_float_maybe(h2v["Total Time (s)"])
    elif "Total (ms)" in h2v:
        ms = _parse_float_maybe(h2v["Total (ms)"])
        total_s = (ms / 1000.0) if ms is not None else None
    elif "Total Time (ms)" in h2v:
        ms = _parse_float_maybe(h2v["Total Time (ms)"])
        total_s = (ms / 1000.0) if ms is not None else None
    elif "Total (s)" in h2v:
        total_s = _parse_float_maybe(h2v["Total (s)"])

    # Throughput
    throughput = None
    for key in ["Throughput (tok/s)", "Throughput", "Tok/s", "Tokens / Second"]:
        if key in h2v:
            throughput = _parse_float_maybe(h2v[key])
            break

    # Optional extras
    prompt_len = _parse_float_maybe(h2v.get("Prompt", "")) if "Prompt" in h2v else None
    avg_gen = _parse_float_maybe(h2v.get("Avg Gen", "")) if "Avg Gen" in h2v else None

    if ttft is None or total_s is None or throughput is None:
        # We require at least these 3 to plot; TPOT can be None
        return None

    out = {
        "Tokens": tokens_val,
        "TTFT (ms)": ttft,
        "TPOT (ms/tok)": tpot if tpot is not None else 0.0,  # keep numeric for plotting; you can change to NaN if preferred
        "Total Time (s)": total_s,
        "Throughput (tok/s)": throughput,
    }
    if prompt_len is not None:
        out["Prompt Len"] = int(round(prompt_len))
    if avg_gen is not None:
        out["Avg Gen"] = avg_gen
    return out

def run_benchmark(tokens: int) -> Optional[Dict]:
    print(f"Running benchmark for {tokens} tokens...")
    
    # Dynamically build the command list
    cmd = RUN_CMD + [str(tokens)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(
            f"Benchmark failed for {tokens} tokens (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )
        return None
    output = result.stdout
    parsed = extract_last_markdown_table(output)
    if not parsed:
        print(f"Error: could not find markdown table in output for {tokens} tokens.\n{output}")
        return None

    header, row = parsed
    metrics = normalize_metrics(header, row, requested_tokens=tokens)
    if not metrics:
        print(f"Error: found table but couldn't normalize metrics for {tokens} tokens.\nHeader={header}\nRow={row}\n\n{output}")
        return None

    # sanity check
    if metrics["Tokens"] != tokens:
        print(f"Warning: parsed Tokens={metrics['Tokens']} != requested {tokens}. Keeping parsed value.")

    return metrics


'''
C++/CUDA version
def run_benchmark(tokens: int) -> Optional[Dict]:
    print(f"Running benchmark for {tokens} tokens...")
    result = subprocess.run([EXECUTABLE, str(tokens)], capture_output=True, text=True)

    if result.returncode != 0:
        print(
            f"Benchmark failed for {tokens} tokens (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )
        return None

    output = result.stdout
    parsed = extract_last_markdown_table(output)
    if not parsed:
        print(f"Error: could not find markdown table in output for {tokens} tokens.\n{output}")
        return None

    header, row = parsed
    metrics = normalize_metrics(header, row, requested_tokens=tokens)
    if not metrics:
        print(f"Error: found table but couldn't normalize metrics for {tokens} tokens.\nHeader={header}\nRow={row}\n\n{output}")
        return None

    # sanity check
    if metrics["Tokens"] != tokens:
        print(f"Warning: parsed Tokens={metrics['Tokens']} != requested {tokens}. Keeping parsed value.")

    return metrics
'''
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

    # Graphs
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("cpu-gpt-engine Performance Metrics (CPU)", fontsize=16)

    axes[0, 0].plot(df["Tokens"], df["TTFT (ms)"], marker="o")
    axes[0, 0].set_title("Time To First Token (TTFT)")
    axes[0, 0].set_xlabel("Tokens Generated")
    axes[0, 0].set_ylabel("Milliseconds")
    axes[0, 0].grid(True)

    axes[0, 1].plot(df["Tokens"], df["TPOT (ms/tok)"], marker="o")
    axes[0, 1].set_title("Time Per Output Token (TPOT)")
    axes[0, 1].set_xlabel("Tokens Generated")
    axes[0, 1].set_ylabel("Milliseconds / Token")
    axes[0, 1].grid(True)
    if (df["TPOT (ms/tok)"] > 0).any():
        axes[0, 1].set_ylim(0, df["TPOT (ms/tok)"].max() * 1.2)

    axes[1, 0].plot(df["Tokens"], df["Total Time (s)"], marker="o")
    axes[1, 0].set_title("Total Execution Time")
    axes[1, 0].set_xlabel("Tokens Generated")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].grid(True)

    axes[1, 1].plot(df["Tokens"], df["Throughput (tok/s)"], marker="o")
    axes[1, 1].set_title("Overall Throughput")
    axes[1, 1].set_xlabel("Tokens Generated")
    axes[1, 1].set_ylabel("Tokens / Second")
    axes[1, 1].grid(True)

    plt.tight_layout()

    out_dir = Path("./benchmarks/hf_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results_pytorch.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nGraphs saved to '{out_path}'")

if __name__ == "__main__":
    main()
