#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
#include "ops.hpp"

#include <cuda_runtime.h>
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

int main(int argc, char** argv) {
    //Setup
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

    // Engine (engine owns stream, cuBLAS handle, LM head, argmax, etc.)
    InferenceEngine engine(model_weights);
    cudaStream_t stream = engine.GetStream();

    //CUDA timing events (recorded on the same stream)
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

        //H2D tokens async on stream
        CudaCheck(cudaMemcpyAsync(d_tokens,
                                  current_input_tokens.data(),
                                  seq_len * (int)sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  stream),
                  "H2D tokens");

        //Forward
        engine.ApplyEmbedding(d_tokens, seq_len, d_input_buffer, current_pos);

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            //LN #1 input
            CudaCheck(cudaMemcpyAsync(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice, stream),
                      "D2D ln_buffer copy #1");
            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 1);

            //Attention
            engine.AttentionLayer(d_ln_buffer, d_attention_output, seq_len, layer_idx, current_pos);
            ops::AddResidual(d_input_buffer, d_attention_output, seq_len * hidden_dim, stream);

            //LN #2 input
            CudaCheck(cudaMemcpyAsync(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice, stream),
                      "D2D ln_buffer copy #2");
            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 2);

            //FFN
            engine.FeedForwardLayer(d_ln_buffer, d_ff_output, d_ff_buffer, seq_len, layer_idx);
            ops::AddResidual(d_input_buffer, d_ff_output, seq_len * hidden_dim, stream);
        }

        float* d_last_token_vec = d_input_buffer + ((seq_len - 1) * hidden_dim);
        engine.ApplyLayerNorm(d_last_token_vec, 1, 0, 3);

        //Stop TTFT after step 0 (prefill)
        if (!prefill_recorded) {
            CudaCheck(cudaEventRecord(stop_prefill, stream), "event record stop_prefill");
            prefill_recorded = true;
        }

        // GPU LM head + GPU argmax (engine-internal)
        int best_token_id = engine.SampleNextToken(d_last_token_vec);

        generated_count++;
        if (best_token_id == eos_id) break;

        current_pos += seq_len;
        current_input_tokens = { best_token_id };
    }

    CudaCheck(cudaEventRecord(stop_total, stream), "event record stop_total");
    CudaCheck(cudaEventSynchronize(stop_total), "event sync stop_total");

    //Metrics
    float ms_total = 0.0f, ms_prefill = 0.0f;
    CudaCheck(cudaEventElapsedTime(&ms_total, start_total, stop_total), "elapsed total");
    CudaCheck(cudaEventElapsedTime(&ms_prefill, start_total, stop_prefill), "elapsed prefill");

    float ms_decode_total = ms_total - ms_prefill;
    int decode_tokens = generated_count - 1;

    float ttft = ms_prefill;
    float tpot = (decode_tokens > 0) ? (ms_decode_total / decode_tokens) : 0.0f;
    float total_s = ms_total / 1000.0f;
    float throughput = (ms_total > 0.0f) ? (generated_count / total_s) : 0.0f;

    std::cout << "\n==================================================\n";
    std::cout << " BENCHMARK RESULTS TABLE\n";
    std::cout << "==================================================\n";
    std::cout << "|   Tokens |   TTFT (ms) |   TPOT (ms/tok) |   Total Time (s) |   Throughput (tok/s) |\n";
    std::cout << "|---------:|------------:|----------------:|-----------------:|---------------------:|\n";
    std::cout << "| " << std::setw(8) << tokens_to_generate
              << " | " << std::setw(11) << std::fixed << std::setprecision(2) << ttft
              << " | " << std::setw(14) << std::fixed << std::setprecision(2) << tpot
              << " | " << std::setw(15) << std::fixed << std::setprecision(2) << total_s
              << " | " << std::setw(20) << std::fixed << std::setprecision(2) << throughput
              << " |\n";

    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_input_buffer);
    cudaFree(d_ln_buffer);
    cudaFree(d_attention_output);
    cudaFree(d_ff_output);
    cudaFree(d_ff_buffer);

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(stop_prefill);

    return 0;
}