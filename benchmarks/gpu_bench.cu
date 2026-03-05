#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
#include "ops.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

int main() {
    // 1. Setup
    std::string weights_file = "data/gpt2_embeddings.bin";
    auto weights = WeightsLoader::load_weights(weights_file);
    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    std::string vocab_path  = "data/vocab.json";
    std::string merges_path = "data/merges.txt";
    Tokenizer tokenizer(vocab_path, merges_path);

    std::string input = "The quick brown fox jumps over the lazy dog and runs into the forest";
    int tokens_to_generate = 20;

    std::vector<int> tokens = tokenizer.Encoder(input);
    int initial_prompt_size = tokens.size();

    int max_seq_len = kMaxSequence; 
    int hidden_dim = kModelSize;

    int* d_tokens = nullptr;
    cudaMalloc((void**)&d_tokens, max_seq_len * sizeof(int));

    float *d_input_buffer, *d_ln_buffer, *d_attention_output, *d_ff_output, *d_ff_buffer;
    cudaMalloc((void**)&d_input_buffer, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_ln_buffer, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_attention_output, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_ff_output, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_ff_buffer, max_seq_len * hidden_dim * 4 * sizeof(float));

    InferenceEngine engine(model_weights);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

    for (int step = 0; step < tokens_to_generate; ++step) {
        int seq_len = tokens.size();
        cudaMemcpy(d_tokens, tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);
        
        engine.ApplyEmbedding(d_tokens, seq_len, d_input_buffer);
        int buffer_bytes = seq_len * hidden_dim * sizeof(float);

        for (size_t layer_idx = 0; layer_idx < kNumLayers; ++layer_idx) {
            cudaMemcpy(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice);
            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 1);
            engine.AttentionLayer(d_ln_buffer, d_attention_output, seq_len, layer_idx);
            ops::AddResidual(d_input_buffer, d_attention_output, seq_len * hidden_dim);
            
            cudaMemcpy(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice);
            engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 2);
            engine.FeedForwardLayer(d_ln_buffer, d_ff_output, d_ff_buffer, seq_len, layer_idx);
            ops::AddResidual(d_input_buffer, d_ff_output, seq_len * hidden_dim);
        }

        float* d_last_token_vec = d_input_buffer + ((seq_len - 1) * hidden_dim);
        engine.ApplyLayerNorm(d_last_token_vec, 1, 0, 3);

        std::vector<float> final_ln_out(hidden_dim);
        cudaMemcpy(final_ln_out.data(), d_last_token_vec, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> logits(kVocabSize);
        for (size_t v = 0; v < kVocabSize; ++v){
            float* vocab_row = model_weights.weight_token_emb + (v * hidden_dim);
            logits[v] = ops::DotProd(final_ln_out.data(), vocab_row, hidden_dim);
        }

        int best_token_id = 0;
        float max_logit = logits[0];
        for (size_t v = 1; v < kVocabSize; ++v){
            if (logits[v] > max_logit){
                max_logit = logits[v];
                best_token_id = v;
            }
        }

        tokens.push_back(best_token_id);
        if (best_token_id == 50256) break; // End of text
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    int generated_tokens = tokens.size() - initial_prompt_size;
    float seconds = milliseconds / 1000.0f;
    float tokens_per_second = generated_tokens / seconds;

    std::cout << "Total execution time: " << seconds << " seconds" << std::endl;
    std::cout << "Tokens generated:     " << generated_tokens << std::endl;
    std::cout << "Throughput:           " << std::fixed << std::setprecision(2) << tokens_per_second << " tok/s" << std::endl;
    std::cout << "Avg time per token:   " << (milliseconds / generated_tokens) << " ms/tok" << std::endl;

    // Cleanup
    cudaFree(d_tokens); cudaFree(d_input_buffer); cudaFree(d_ln_buffer);
    cudaFree(d_attention_output); cudaFree(d_ff_output); cudaFree(d_ff_buffer);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
