#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
#include "ops.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>

static inline void CudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

struct RunStats {
    float ms_total = 0.0f;
    float ms_ttft  = 0.0f;
    int generated  = 0; // number of generated tokens (including first)
};

static RunStats RunOnce(
    InferenceEngine& engine,
    cudaStream_t stream,
    const std::vector<int>& prompt_tokens,
    int tokens_to_generate,
    int* d_tokens,
    int max_seq_len,
    int hidden_dim,
    int num_layers,
    int eos_id,
    float* d_input_buffer,
    float* d_ln_buffer,
    float* d_attention_output,
    float* d_ff_output,
    float* d_ff_buffer,
    int* h_tokens_pinned  // pinned staging buffer for H2D
) {
    // events
    cudaEvent_t start_total, stop_total, stop_ttft;
    CudaCheck(cudaEventCreate(&start_total), "event create start_total");
    CudaCheck(cudaEventCreate(&stop_total),  "event create stop_total");
    CudaCheck(cudaEventCreate(&stop_ttft),   "event create stop_ttft");

    engine.ResetKV();
    std::vector<int> current_input_tokens = prompt_tokens;
    int current_pos = 0;
    int generated_count = 0;
    bool ttft_recorded = false;

    CudaCheck(cudaEventRecord(start_total, stream), "event record start_total");

    for (int step = 0; step < tokens_to_generate; ++step) {
        int seq_len = (int)current_input_tokens.size();
        int buffer_bytes = seq_len * hidden_dim * (int)sizeof(float);

        // copy into pinned staging (host)
        for (int i = 0; i < seq_len; ++i) h_tokens_pinned[i] = current_input_tokens[i];

        // H2D async from pinned host
        CudaCheck(cudaMemcpyAsync(d_tokens,
                                  h_tokens_pinned,
                                  seq_len * (int)sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  stream),
                  "H2D tokens");

        engine.ApplyEmbedding(d_tokens, seq_len, d_input_buffer, current_pos);

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            // NOTE: these D2D copies distort perf; best is to remove by changing LN API.
            CudaCheck(cudaMemcpyAsync(d_ln_buffer, d_input_buffer, buffer_bytes,
                                      cudaMemcpyDeviceToDevice, stream),
                      "D2D ln_buffer copy #1");
            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 1);

            engine.AttentionLayer(d_ln_buffer, d_attention_output, seq_len, layer_idx, current_pos);
            ops::AddResidual(d_input_buffer, d_attention_output, seq_len * hidden_dim, stream);

            CudaCheck(cudaMemcpyAsync(d_ln_buffer, d_input_buffer, buffer_bytes,
                                      cudaMemcpyDeviceToDevice, stream),
                      "D2D ln_buffer copy #2");
            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 2);

            engine.FeedForwardLayer(d_ln_buffer, d_ff_output, d_ff_buffer, seq_len, layer_idx);
            ops::AddResidual(d_input_buffer, d_ff_output, seq_len * hidden_dim, stream);
        }

        float* d_last_token_vec = d_input_buffer + ((seq_len - 1) * hidden_dim);
        engine.ApplyLayerNorm(d_last_token_vec, 1, 0, 3);

        // Sample (in BENCHMARK_MODE this should NOT stream-sync per token)
        int best_token_id = engine.SampleNextToken(d_last_token_vec);
        generated_count++;

        // TTFT: time until first generated token is available (on GPU timeline)
        if (!ttft_recorded) {
            CudaCheck(cudaEventRecord(stop_ttft, stream), "event record stop_ttft");
            ttft_recorded = true;
        }

        if (best_token_id == eos_id) break;

        current_pos += seq_len;
        current_input_tokens = { best_token_id };
    }

    CudaCheck(cudaEventRecord(stop_total, stream), "event record stop_total");
    CudaCheck(cudaEventSynchronize(stop_total), "event sync stop_total");

    RunStats out;
    CudaCheck(cudaEventElapsedTime(&out.ms_total, start_total, stop_total), "elapsed total");
    CudaCheck(cudaEventElapsedTime(&out.ms_ttft,  start_total, stop_ttft),  "elapsed ttft");
    out.generated = generated_count;

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(stop_ttft);
    return out;
}

int main(int argc, char** argv) {
    std::string weights_file = "data/gpt2_embeddings.bin";
    auto weights = WeightsLoader::load_weights(weights_file);
    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    std::string vocab_path  = "data/vocab.json";
    std::string merges_path = "data/merges.txt";
    Tokenizer tokenizer(vocab_path, merges_path);

    std::string input = "The quick brown fox jumps over the lazy dog and runs into the forest";
    int tokens_to_generate = 128;               // better default for stable TPOT
    if (argc > 1) tokens_to_generate = std::stoi(argv[1]);

    std::vector<int> prompt_tokens = tokenizer.Encoder(input);
    int prompt_len = (int)prompt_tokens.size();

    const int max_seq_len = 1024;
    const int hidden_dim  = 768;
    const int num_layers  = 12;
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

    // pinned host token staging
    int* h_tokens_pinned = nullptr;
    CudaCheck(cudaMallocHost((void**)&h_tokens_pinned, max_seq_len * sizeof(int)), "cudaMallocHost h_tokens_pinned");

    // Engine
    InferenceEngine engine(model_weights);
    cudaStream_t stream = engine.GetStream();

    // Warmup + measure
    const int WARMUP_ITERS  = 5;
    const int MEASURE_ITERS = 20;

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        engine.ResetKV();
        (void)RunOnce(engine, stream, prompt_tokens, tokens_to_generate,
                      d_tokens, max_seq_len, hidden_dim, num_layers, eos_id,
                      d_input_buffer, d_ln_buffer, d_attention_output, d_ff_output, d_ff_buffer,
                      h_tokens_pinned);
    }

    std::vector<RunStats> runs;
    runs.reserve(MEASURE_ITERS);
    for (int i = 0; i < MEASURE_ITERS; ++i) {
        runs.push_back(RunOnce(engine, stream, prompt_tokens, tokens_to_generate,
                               d_tokens, max_seq_len, hidden_dim, num_layers, eos_id,
                               d_input_buffer, d_ln_buffer, d_attention_output, d_ff_output, d_ff_buffer,
                               h_tokens_pinned));
    }

    // aggregate (mean)
    float mean_total = 0.0f, mean_ttft = 0.0f;
    float mean_tpot  = 0.0f;
    float mean_throughput = 0.0f;

    for (const auto& r : runs) {
        mean_total += r.ms_total;
        mean_ttft  += r.ms_ttft;

        int decode_tokens = r.generated - 1;
        float ms_decode = r.ms_total - r.ms_ttft;
        float tpot = (decode_tokens > 0) ? (ms_decode / decode_tokens) : 0.0f;
        mean_tpot += tpot;

        float total_s = r.ms_total / 1000.0f;
        float thr = (total_s > 0) ? (r.generated / total_s) : 0.0f;
        mean_throughput += thr;
    }

    mean_total /= (float)runs.size();
    mean_ttft  /= (float)runs.size();
    mean_tpot  /= (float)runs.size();
    mean_throughput /= (float)runs.size();

    std::cout << "\n==================================================\n";
    std::cout << " GPU BENCHMARK (mean over " << MEASURE_ITERS << " runs, warmup " << WARMUP_ITERS << ")\n";
    std::cout << " prompt_len=" << prompt_len << ", requested_gen=" << tokens_to_generate << "\n";
    std::cout << "==================================================\n";
    std::cout << "| Prompt | Req Gen |   TTFT (ms) |   TPOT (ms/tok) |   Total (ms) |   Throughput (tok/s) |\n";
    std::cout << "|-------:|--------:|------------:|----------------:|-------------:|---------------------:|\n";
    std::cout << "| " << std::setw(6) << prompt_len
              << " | " << std::setw(7) << tokens_to_generate
              << " | " << std::setw(11) << std::fixed << std::setprecision(2) << mean_ttft
              << " | " << std::setw(14) << std::fixed << std::setprecision(2) << mean_tpot
              << " | " << std::setw(11) << std::fixed << std::setprecision(2) << mean_total
              << " | " << std::setw(20) << std::fixed << std::setprecision(2) << mean_throughput
              << " |\n";

    cudaFreeHost(h_tokens_pinned);

    cudaFree(d_tokens);
    cudaFree(d_input_buffer);
    cudaFree(d_ln_buffer);
    cudaFree(d_attention_output);
    cudaFree(d_ff_output);
    cudaFree(d_ff_buffer);

    return 0;
}
