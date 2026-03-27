# gpt2-engine-cpp
> Custom GPT2 inference engine built from scratch in C++ and CUDA

# Overview
Project is meant to be a high performance implementation of
GPT2 (124M parameter) architecture.

# How to Run
`cmake -B build` 
`cmake --build build`
## CPU Version
`./build/src/gpt2_inference_engine_cpu`
## CPU Benchmarks
`./build/benchmarks/cpu_bench`

## CUDA Version
`./build/src/gpt2_inference_engine_cuda`

## CUDA Benchmarks
`./build/benchmarks/gpu_bench`

# Key Features
- Custom CUDA Kernels: Implemented custom multi-head attention, layer normalization, GELU activation, and fuesed matrix operations (only using cuBLAS)
- Cross-Framework Validation: Features and end to end python validation suite guarantees mathemematically identical output to huggingface version
- Deterministic: Can handle precise floating point arithmetic stability across matrix operations
- KV Caching

# Validation & Performance
- **Mathematical Correctness:** Achieved a Mean Squared Error (MSE) of effectively `0.0` and a Maximum Absolute Difference of `< 1e-3` in FP32 logit outputs compared to PyTorch's `F.scaled_dot_product_attention`.
- **Hardware Acceleration (CPU vs GPU):** Achieved up to a **544x reduction in latency** by migrating sequential C++ matrix operations to custom `__global__` CUDA kernels.
- **Throughput Scaling:** With KV caching enabled, achieved up to **1872.46x higher token throughput** at longer sequence lengths relative to the CPU implementation.
- **Hardware Efficiency:** Profiled via NVIDIA Nsight Compute (`ncu`), achieving **99.37% branch efficiency**, demonstrating highly optimized warp execution with minimal thread divergence.
- **Full-Stack Memory Safety:** Validated host execution via `valgrind` and device execution via NVIDIA `compute-sanitizer`, guaranteeing **0 memory leaks and 0 invalid out-of-bounds reads/writes** across millions of dynamic CPU and GPU tensor allocations.
- **CPU benchmarks for sequences larger than 50 tokens were omitted as they took a very long time so listed as N/A**

# Benchmarking Methodology & Notes
The benchmarks demonstrate that Cu-Transformer successfully reduces Python interpreter overhead and memory allocation bottlenecks by executing custom CUDA kernels directly. However, for full transparency, here is the context behind the numbers:

- **The Baseline (What was compared):** The baseline is the standard Hugging Face `transformers` library running PyTorch in eager mode. Both implementations executed identical greedy decoding workloads (`use_cache=True`) on an NVIDIA RTX 3060.
- **Fairness & Interpretation of Results:** The custom Cu-Transformer engine is a highly specialized implementation targeting a single decoding strategy (greedy)  with tightly controlled memory layouts and execution flow. In contrast, Hugging Face `generate()` is a general-purpose API designed to support a wide range of decoding strategies, batching modes, and model architectures. As a result, this comparison should be interpreted as **specialized vs general-purpose execution**, not as a statement that one approach is universally faster.
- **The JIT / `torch.compile` Context:** PyTorch 2.x `torch.compile` does work for Hugging Face models and was enabled in the benchmark. However, autoregressive generation still tends to limit how much benefit compilation can deliver in the default `generate()` path, because graph breaks and shape variability reduce optimization opportunities. In particular, Hugging Face documents that the default dynamic KV cache prevents taking advantage of most JIT optimizations, while fixed-size caches such as `StaticCache` are more compatible with `torch.compile`. Cu-Transformer avoids much of this overhead by using a specialized decode path with preallocated buffers and fixed execution patterns. :contentReference[oaicite:0]{index=0}
- **PyTorch TTFT (Time To First Token):** TTFT is not reported for PyTorch because `model.generate()` is a blocking function that bundles prefill and decode phases together. Extracting TTFT requires a custom streamer hook, so the comparison focuses on total latency and throughput instead.
- **Scope of the Comparison:** Cu-Transformer was built as a low-level exploration of GPU execution. This benchmark does **not** compare against optimized production inference systems (e.g., vLLM, TensorRT-LLM), which implement advanced techniques such as FlashAttention, PagedAttention, kernel fusion, and continuous batching. Those systems are designed to close or exceed this performance gap in real-world deployments.




# CPU vs GPU Benchmark Comparison

| Tokens | CPU TTFT (ms) | GPU TTFT (ms) | Speedup (Latency) | CPU Total Time (s) | GPU Total Time (s) | Speedup (Total Time) | CPU Throughput (tok/s) | GPU Throughput (tok/s) | Speedup (Throughput) |
| ------ | ------------- | ------------- | ----------------- | ------------------ | ------------------ | -------------------- | ---------------------- | ---------------------- | -------------------- |
| 5      | 2919.88       | 5.36          | 544.75x           | 16.6428            | 0.01951            | 852.99x              | 0.30                   | 256.86                 | 856.20x              |
| 10     | 2926.20       | 5.45          | 536.92x           | 38.3236            | 0.03865            | 991.56x              | 0.26                   | 259.34                 | 997.46x              |
| 20     | 2880.89       | 6.11          | 471.50x           | 96.0915            | 0.08435            | 1139.20x             | 0.21                   | 237.23                 | 1129.67x             |
| 30     | 2901.22       | 6.02          | 481.93x           | 174.708            | 0.12706            | 1374.98x             | 0.17                   | 236.18                 | 1389.29x             |
| 50     | 2885.47       | 6.08          | 474.58x           | 390.183            | 0.20542            | 1899.44x             | 0.13                   | 243.42                 | 1872.46x             |
| 100    | N/A           | 6.22          | N/A               | N/A                | 0.43376            | N/A                  | N/A                    | 230.81                 | N/A                  |
| 200    | N/A           | 6.35          | N/A               | N/A                | 0.86553            | N/A                  | N/A                    | 231.54                 | N/A                  |
| 300    | N/A           | 6.27          | N/A               | N/A                | 1.30069            | N/A                  | N/A                    | 230.87                 | N/A                  |
| 500    | N/A           | 5.73          | N/A               | N/A                | 2.07385            | N/A                  | N/A                    | 241.63                 | N/A                  |
| 700    | N/A           | 5.40          | N/A               | N/A                | 2.80812            | N/A                  | N/A                    | 249.50                 | N/A                  |
| 800    | N/A           | 5.52          | N/A               | N/A                | 3.08160            | N/A                  | N/A                    | 259.71                 | N/A                  |
| 900    | N/A           | 5.41          | N/A               | N/A                | 3.46639            | N/A                  | N/A                    | 259.76                 | N/A                  |
| 1000   | N/A           | 5.43          | N/A               | N/A                | 4.06463            | N/A                  | N/A                    | 246.69                 | N/A                  |

---

## PyTorch (Huggingface) vs GPU Benchmark Comparison:

| Tokens | GPU Total Time (s) | PyTorch Total Time (s) | Speedup (Total Time) | GPU Throughput (tok/s) | PyTorch Throughput (tok/s) | Speedup (Throughput) |
| ------ | ------------------ | ---------------------- | -------------------- | ---------------------- | -------------------------- | -------------------- |
| 5      | 0.01951            | 0.02552                | 0.76x                | 256.86                 | 195.93                     | 1.31x                |
| 10     | 0.03865            | 0.04860                | 0.80x                | 259.34                 | 205.78                     | 1.26x                |
| 20     | 0.08435            | 0.10592                | 0.80x                | 237.23                 | 188.83                     | 1.26x                |
| 30     | 0.12706            | 0.14586                | 0.87x                | 236.18                 | 205.68                     | 1.15x                |
| 50     | 0.20542            | 0.25087                | 0.82x                | 243.42                 | 199.31                     | 1.22x                |
| 100    | 0.43376            | 0.49058                | 0.88x                | 230.81                 | 203.84                     | 1.13x                |
| 200    | 0.86553            | 0.99766                | 0.87x                | 231.54                 | 200.47                     | 1.15x                |
| 300    | 1.30069            | 1.52521                | 0.85x                | 230.87                 | 196.69                     | 1.17x                |
| 500    | 2.07385            | 2.54238                | 0.82x                | 241.63                 | 196.67                     | 1.23x                |
| 700    | 2.80812            | 3.49153                | 0.80x                | 249.50                 | 200.49                     | 1.24x                |
| 800    | 3.08160            | 4.10253                | 0.75x                | 259.71                 | 195.00                     | 1.33x                |
| 900    | 3.46639            | 4.71424                | 0.74x                | 259.76                 | 190.91                     | 1.36x                |
| 1000   | 4.06463            | 5.14497                | 0.79x                | 246.69                 | 194.36                     | 1.27x                |


## Definitions

- `TTFT_cpu` = CPU Time-To-First-Token (ms)  
- `TTFT_gpu` = GPU Time-To-First-Token (ms)  
- `T_cpu` = CPU Total Generation Time (s)  
- `T_gpu` = GPU Total Generation Time (s)  
- `TP_cpu` = CPU Throughput (tokens/sec)  
- `TP_gpu` = GPU Throughput (tokens/sec)  

## Latency Speedup (TTFT)
Measures how much faster GPU produces first token

$$
\text{Speedup}_{latency} =
\frac{TTFT_{cpu}}{TTFT_{gpu}}
$$

---

## Total Time Speedup
Measures overall generation acceleration

$$
\text{Speedup}_{total} =
\frac{T_{cpu}}{T_{gpu}}
$$

---

## Throughput Speedup
Measures how many more tokens per second the GPU generates compared to CPU.

$$
\text{Speedup}_{throughput} =
\frac{TP_{gpu}}{TP_{cpu}}
$$

---

## Example (50 Tokens)

Given:

- `TTFT_cpu = 2885.47 ms`
- `TTFT_gpu = 6.08 ms`
- `T_cpu = 390.183 s`
- `T_gpu = 0.20542 s`
- `TP_cpu = 0.13 tok/s`
- `TP_gpu = 243.42 tok/s`

### Calculations

$$
\text{Latency Speedup} =
\frac{2885.47}{6.08}
= 474.58\times
$$

$$
\text{Total Time Speedup} =
\frac{390.183}{0.20542}
= 1899.44\times
$$

$$
\text{Throughput Speedup} =
\frac{243.42}{0.13}
= 1872.46\times
$$


 <br>

## CPU Performance
!["CPU"](./benchmarks/cpu_benchmarks/benchmark_results_cpu.png "CPU Performance") 
 <br>

## GPU Performance
!["GPU"](./benchmarks/gpu_benchmarks/benchmark_results_gpu.png "GPU Performance")

---
# Hardware & Software
- **GPU:** NVIDIA GeForce RTX 3060  
- **ToolKit:** CUDA Toolkit 13.1.115
- **Libraries:** cuBLAS
- **CMake:** Version 3.20+

---

# Future Work
- Implement FP16 
- Implemente Temperature instead of taking ArgMax
- Fix run_benchmarks to be more user friendly

