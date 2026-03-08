// gpu_bench.cu (drop-in replacement)

#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
#include "ops.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

static inline void CudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

// 1-block argmax for ~50k logits (good enough for benchmark)
__global__ void ArgMaxKernel(const float* logits, int n, int* out_idx) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    int tid = threadIdx.x;
    float best_v = -1e30f;
    int best_i = 0;

    for (int i = tid; i < n; i += blockDim.x) {
        float v = logits[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }

    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float v2 = s_val[tid + offset];
            int   i2 = s_idx[tid + offset];
            if (v2 > s_val[tid]) {
                s_val[tid] = v2;
                s_idx[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) *out_idx = s_idx[0];
}

int main(int argc, char** argv) {
    // 1) Setup
    std::string weights_file = "data/gpt2_embeddings.bin";
    auto weights = WeightsLoader::load_weights(weights_file);
    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    std::string vocab_path  = "data/vocab.json";
    std::string merges_path = "data/merges.txt";
    Tokenizer tokenizer(vocab_path, merges_path);

    std::string input = "The quick brown fox jumps over the lazy dog and runs into the forest";
    int tokens_to_generate = 500;
    if (argc > 1) tokens_to_generate = std::stoi(argv[1]);

    std::vector<int> tokens = tokenizer.Encoder(input);
    int initial_prompt_size = (int)tokens.size();

    const int max_seq_len = 1024;
    const int hidden_dim  = 768;
    const int num_layers  = 12;
    const int vocab_size  = 50257;
    const int eos_id      = 50256;

    // Device buffers
    int* d_tokens = nullptr;
    CudaCheck(cudaMalloc((void**)&d_tokens, max_seq_len * sizeof(int)), "malloc d_tokens");

    float *d_input_buffer=nullptr, *d_ln_buffer=nullptr, *d_attention_output=nullptr, *d_ff_output=nullptr, *d_ff_buffer=nullptr;
    CudaCheck(cudaMalloc((void**)&d_input_buffer,     max_seq_len * hidden_dim * sizeof(float)), "malloc d_input_buffer");
    CudaCheck(cudaMalloc((void**)&d_ln_buffer,        max_seq_len * hidden_dim * sizeof(float)), "malloc d_ln_buffer");
    CudaCheck(cudaMalloc((void**)&d_attention_output, max_seq_len * hidden_dim * sizeof(float)), "malloc d_attention_output");
    CudaCheck(cudaMalloc((void**)&d_ff_output,        max_seq_len * hidden_dim * sizeof(float)), "malloc d_ff_output");
    CudaCheck(cudaMalloc((void**)&d_ff_buffer,        max_seq_len * hidden_dim * 4 * sizeof(float)), "malloc d_ff_buffer");

    // Engine (assumes your engine creates a stream_ and sets cuBLAS handle to it)
    InferenceEngine engine(model_weights);
    cudaStream_t stream = engine.GetStream();
    cublasHandle_t handle = engine.GetCublasHandle();

    // ---- GPU LM head: copy token embedding matrix to GPU once ----
    // GPT-2 ties output projection to token embeddings, so W = token_emb
    float* d_token_emb = nullptr;
    CudaCheck(cudaMalloc((void**)&d_token_emb, vocab_size * hidden_dim * sizeof(float)), "malloc d_token_emb");
    CudaCheck(cudaMemcpyAsync(d_token_emb,
                              model_weights.weight_token_emb,
                              vocab_size * hidden_dim * sizeof(float),
                              cudaMemcpyHostToDevice,
                              stream),
              "H2D token_emb");

    float* d_logits = nullptr;
    int* d_best_token = nullptr;
    CudaCheck(cudaMalloc((void**)&d_logits, vocab_size * sizeof(float)), "malloc d_logits");
    CudaCheck(cudaMalloc((void**)&d_best_token, sizeof(int)), "malloc d_best_token");

    // 2) CUDA timing events (recorded on the same stream)
    cudaEvent_t start_total, stop_total, stop_prefill;
    CudaCheck(cudaEventCreate(&start_total), "event create start_total");
    CudaCheck(cudaEventCreate(&stop_total), "event create stop_total");
    CudaCheck(cudaEventCreate(&stop_prefill), "event create stop_prefill");

    std::vector<int> current_input_tokens = tokens;
    int current_pos = 0;
    int generated_count = 0;

    CudaCheck(cudaEventRecord(start_total, stream), "event record start_total");
    bool prefill_recorded = false;

    for (int step = 0; step < tokens_to_generate; ++step) {
        int seq_len = (int)current_input_tokens.size();
        int buffer_bytes = seq_len * hidden_dim * (int)sizeof(float);

        // H2D tokens async on stream
        CudaCheck(cudaMemcpyAsync(d_tokens,
                                  current_input_tokens.data(),
                                  seq_len * (int)sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  stream),
                  "H2D tokens");

        // Forward
        engine.ApplyEmbedding(d_tokens, seq_len, d_input_buffer, current_pos);

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            CudaCheck(cudaMemcpyAsync(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice, stream),
                      "D2D ln_buffer copy");

            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 1);
            engine.AttentionLayer(d_ln_buffer, d_attention_output, seq_len, layer_idx, current_pos);

            ops::AddResidual(d_input_buffer, d_attention_output, seq_len * hidden_dim, stream);

            CudaCheck(cudaMemcpyAsync(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice, stream),
                      "D2D ln_buffer copy 2");

            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 2);
            engine.FeedForwardLayer(d_ln_buffer, d_ff_output, d_ff_buffer, seq_len, layer_idx);

            ops::AddResidual(d_input_buffer, d_ff_output, seq_len * hidden_dim, stream);
        }

        float* d_last_token_vec = d_input_buffer + ((seq_len - 1) * hidden_dim);
        engine.ApplyLayerNorm(d_last_token_vec, 1, 0, 3);

        // Stop TTFT after step 0 (prefill)
        if (!prefill_recorded) {
            CudaCheck(cudaEventRecord(stop_prefill, stream), "event record stop_prefill");
            prefill_recorded = true;
        }

        // ---- GPU logits + GPU argmax ----
        // logits(1 x vocab) = hidden(1 x hidden) * token_emb^T(hidden x vocab)
        ops::MatMulTransposedB(handle, d_last_token_vec, d_token_emb, d_logits, /*M=*/1, /*N=*/vocab_size, /*K=*/hidden_dim);

        ArgMaxKernel<<<1, 256, 0, stream>>>(d_logits, vocab_size, d_best_token);
        CudaCheck(cudaPeekAtLastError(), "ArgMaxKernel launch");

        int best_token_id = 0;
        CudaCheck(cudaMemcpyAsync(&best_token_id, d_best_token, sizeof(int), cudaMemcpyDeviceToHost, stream),
                  "D2H best_token_id");
        CudaCheck(cudaStreamSynchronize(stream), "sync for best_token_id");

        generated_count++;
        if (best_token_id == eos_id) break;

        // next step is decode (1 token)
        current_pos += seq_len;
        current_input_tokens = { best_token_id };
    }

    CudaCheck(cudaEventRecord(stop_total, stream), "event record stop_total");
    CudaCheck(cudaEventSynchronize(stop_total), "event sync stop_total");

    // 3) Metrics
    float ms_total = 0.0f, ms_prefill = 0.0f;
    CudaCheck(cudaEventElapsedTime(&ms_total, start_total, stop_total), "elapsed total");
    CudaCheck(cudaEventElapsedTime(&ms_prefill, start_total, stop_prefill), "elapsed prefill");

    float ms_decode_total = ms_total - ms_prefill;
    int decode_tokens = generated_count - 1;

    float ttft = ms_prefill;
    float tpot = (decode_tokens > 0) ? (ms_decode_total / decode_tokens) : 0.0f;
    float overall_throughput = (generated_count > 0) ? (generated_count / (ms_total / 1000.0f)) : 0.0f;

    std::cout << "\n========== KV CACHE BENCHMARK ==========\n";
    std::cout << "Input prompt size:    " << initial_prompt_size << " tokens\n";
    std::cout << "Tokens generated:     " << generated_count << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "TTFT (Prefill):       " << std::fixed << std::setprecision(2) << ttft << " ms\n";
    std::cout << "TPOT (Decode Avg):    " << tpot << " ms/tok\n";
    std::cout << "Total execution time: " << (ms_total / 1000.0f) << " seconds\n";
    std::cout << "Overall Throughput:   " << overall_throughput << " tok/s\n";
    std::cout << "========================================\n";

    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_input_buffer);
    cudaFree(d_ln_buffer);
    cudaFree(d_attention_output);
    cudaFree(d_ff_output);
    cudaFree(d_ff_buffer);
    cudaFree(d_token_emb);
    cudaFree(d_logits);
    cudaFree(d_best_token);

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(stop_prefill);

    return 0;
}
