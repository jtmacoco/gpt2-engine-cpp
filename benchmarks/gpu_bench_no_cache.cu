#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
#include "ops.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

static inline void CUDA_OK(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

int main() {
    // 1) Setup
    std::string weights_file = "data/gpt2_embeddings.bin";
    auto weights = WeightsLoader::load_weights(weights_file);
    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    std::string vocab_path  = "data/vocab.json";
    std::string merges_path = "data/merges.txt";
    Tokenizer tokenizer(vocab_path, merges_path);

    std::string input = "The quick brown fox jumps over the lazy dog and runs into the forest";
    int tokens_to_generate = 200; // change as desired

    std::vector<int> tokens = tokenizer.Encoder(input);
    int initial_prompt_size = (int)tokens.size();

    const int max_seq_len = kMaxSequence;
    const int hidden_dim  = kModelSize;

    // 2) Device buffers
    int* d_tokens = nullptr;
    CUDA_OK(cudaMalloc((void**)&d_tokens, max_seq_len * sizeof(int)), "cudaMalloc d_tokens");

    float *d_input_buffer=nullptr, *d_ln_buffer=nullptr, *d_attention_output=nullptr, *d_ff_output=nullptr, *d_ff_buffer=nullptr;
    CUDA_OK(cudaMalloc((void**)&d_input_buffer,     max_seq_len * hidden_dim * sizeof(float)), "cudaMalloc d_input_buffer");
    CUDA_OK(cudaMalloc((void**)&d_ln_buffer,        max_seq_len * hidden_dim * sizeof(float)), "cudaMalloc d_ln_buffer");
    CUDA_OK(cudaMalloc((void**)&d_attention_output, max_seq_len * hidden_dim * sizeof(float)), "cudaMalloc d_attention_output");
    CUDA_OK(cudaMalloc((void**)&d_ff_output,        max_seq_len * hidden_dim * sizeof(float)), "cudaMalloc d_ff_output");
    CUDA_OK(cudaMalloc((void**)&d_ff_buffer,        max_seq_len * hidden_dim * 4 * sizeof(float)), "cudaMalloc d_ff_buffer");

    // 3) Engine + stream
    InferenceEngine engine(model_weights);
    cudaStream_t stream = engine.GetStream();

    // 4) Timing events (record on the engine stream)
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start), "cudaEventCreate start");
    CUDA_OK(cudaEventCreate(&stop),  "cudaEventCreate stop");

    CUDA_OK(cudaEventRecord(start, stream), "cudaEventRecord start");

    int current_pos = 0;
    int generated_tokens = 0;

    // We'll use a "current_input_tokens" vector like your main code:
    std::vector<int> current_input_tokens = tokens;

    for (int step = 0; step < tokens_to_generate; ++step) {
        int seq_len = (int)current_input_tokens.size();
        if (seq_len <= 0) break;

        // H2D tokens on the engine stream
        CUDA_OK(cudaMemcpyAsync(d_tokens,
                                current_input_tokens.data(),
                                seq_len * (int)sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync tokens H2D");

        // Embedding (uses current_pos for absolute positions)
        engine.ApplyEmbedding(d_tokens, seq_len, d_input_buffer, current_pos);

        int buffer_bytes = seq_len * hidden_dim * (int)sizeof(float);

        for (int layer_idx = 0; layer_idx < (int)kNumLayers; ++layer_idx) {
            // LN #1
            CUDA_OK(cudaMemcpyAsync(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpyAsync D2D ln_buffer #1");
            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 1);

            // Attention (KV-cache)
            engine.AttentionLayer(d_ln_buffer, d_attention_output, seq_len, layer_idx, current_pos);
            ops::AddResidual(d_input_buffer, d_attention_output, seq_len * hidden_dim, stream);

            // LN #2
            CUDA_OK(cudaMemcpyAsync(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpyAsync D2D ln_buffer #2");
            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 2);

            // FFN
            engine.FeedForwardLayer(d_ln_buffer, d_ff_output, d_ff_buffer, seq_len, layer_idx);
            ops::AddResidual(d_input_buffer, d_ff_output, seq_len * hidden_dim, stream);
        }

        // Final LN on last token
        float* d_last_token_vec = d_input_buffer + ((seq_len - 1) * hidden_dim);
        engine.ApplyLayerNorm(d_last_token_vec, 1, 0, 3);

        // GPU vocab projection + GPU argmax (returns token id to CPU)
        int best_token_id = engine.SampleNextToken(d_last_token_vec);

        // Update for next iteration (decode: feed only 1 token)
        generated_tokens++;
        if (best_token_id == 50256) break;

        current_pos += seq_len;
        current_input_tokens = { best_token_id };
    }

    CUDA_OK(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    CUDA_OK(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    float seconds = ms / 1000.0f;
    float tps = (seconds > 0.0f) ? (generated_tokens / seconds) : 0.0f;
    float ms_per_tok = (generated_tokens > 0) ? (ms / generated_tokens) : 0.0f;

    std::cout << "Total execution time: " << std::fixed << std::setprecision(4) << seconds << " s\n";
    std::cout << "Prompt tokens:        " << initial_prompt_size << "\n";
    std::cout << "Tokens generated:     " << generated_tokens << "\n";
    std::cout << "Throughput:           " << std::fixed << std::setprecision(2) << tps << " tok/s\n";
    std::cout << "Avg time per token:   " << std::fixed << std::setprecision(2) << ms_per_tok << " ms/tok\n";

    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_input_buffer);
    cudaFree(d_ln_buffer);
    cudaFree(d_attention_output);
    cudaFree(d_ff_output);
    cudaFree(d_ff_buffer);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}