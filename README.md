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

# Validation & Performance
- **Mathematical Correctness:** Achieved a Mean Squared Error (MSE) of effectively `0.0` and a Maximum Absolute Difference of `< 1e-3` in FP32 logit outputs compared to PyTorch's `F.scaled_dot_product_attention`.
- **Hardware Acceleration (CPU vs GPU):** Achieved up to a **71x inference speedup** by migrating sequential C++ matrix operations to custom `__global__` CUDA kernels.
- **Throughput Scaling:** Benchmarked sequence generation degradation to profile the $O(N^2)$ attention bottleneck, establishing the baseline for future KV-Cache implementation:

| Tokens Generated | CPU Baseline | Custom CUDA Engine | Speedup Multiplier |
| :--- | :--- | :--- | :--- |
| **20 Tokens** | 0.21 tok/s | 9.38 tok/s | ~44x |
| **30 Tokens** | 0.18 tok/s | 9.31 tok/s | ~51x |
| **50 Tokens** | 0.13 tok/s | 9.25 tok/s | ~71x |
| **80 Tokens** | N/A | 9.07 tok/s | N/A |
| **100 Tokens** | N/A | 8.97 tok/s | N/A |

- **Hardware Efficiency:** Profiled via NVIDIA Nsight Compute (`ncu`), achieving **99.37% branch efficiency**, demonstrating highly optimized warp execution with minimal thread divergence.
- **Full-Stack Memory Safety:** Validated host execution via `valgrind` and device execution via NVIDIA `compute-sanitizer`, guaranteeing **0 memory leaks and 0 invalid out-of-bounds reads/writes** across millions of dynamic CPU and GPU tensor allocations.
- Reason I didn't do larger tokens after 50 for the CPU version is because it takes to long


# Future Work
- Based on the ncu output KV Caching should be implemented
- View ncu output in `./benchmarks/ncu_reports`

