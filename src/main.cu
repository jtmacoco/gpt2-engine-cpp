#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
#include "ops.hpp"
#include <iostream>
#include <filesystem>
#include <string>
namespace fs = std::filesystem;


int main(int argc, char** argv){
    std::string weights_file = "data/gpt2_embeddings.bin";

    //contains both positional and token weights
    auto weights = WeightsLoader::load_weights(weights_file);

    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    std::string vocab_path  = "data/vocab.json";
    std::string merges_path = "data/merges.txt";

    Tokenizer tokenizer(vocab_path,merges_path);

    std::string input = "The quick brown fox jumps over the lazy";
    int max_tokens_to_generate = 20;
    int generated_count = 0;

    std::vector<int> tokens = tokenizer.Encoder(input);
    std::cout<< "Input: " << input <<std::endl;
    std::vector<int> output_tokens;

    int max_seq_len = kMaxSequence;
    int hidden_dim = kModelSize;

    int* d_tokens = nullptr;
    cudaMalloc((void**)&d_tokens, max_seq_len * sizeof(int));

    float* d_input_buffer = nullptr;
    float* d_ln_buffer = nullptr;
    float* d_attention_output = nullptr;
    float* d_ff_output = nullptr;
    float* d_ff_buffer = nullptr;

    cudaMalloc((void**)&d_input_buffer, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_ln_buffer, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_attention_output, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_ff_output, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_ff_buffer, max_seq_len * hidden_dim * 4 * sizeof(float));

    InferenceEngine inference_engine(model_weights);
    while (generated_count < max_tokens_to_generate){
        int seq_len = tokens.size();

        cudaMemcpy(d_tokens, tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);
        inference_engine.ApplyEmbedding(d_tokens, seq_len, d_input_buffer);

        int buffer_bytes = seq_len * kModelSize * sizeof(float);

        for (size_t layer_idx = 0; layer_idx < kNumLayers; ++layer_idx){
            cudaMemcpy(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice);
            inference_engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 1);
            inference_engine.AttentionLayer(d_ln_buffer, d_attention_output, seq_len, layer_idx);
            ops::AddResidual(d_input_buffer, d_attention_output, seq_len * kModelSize);
            cudaDeviceSynchronize();

            cudaMemcpy(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice);
            inference_engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 2);
            inference_engine.FeedForwardLayer(d_ln_buffer, d_ff_output, d_ff_buffer, seq_len, layer_idx);
            ops::AddResidual(d_input_buffer, d_ff_output, seq_len * kModelSize);
            cudaDeviceSynchronize();
        }

        float* d_last_token_vec = d_input_buffer + ((seq_len - 1) * hidden_dim);

        inference_engine.ApplyLayerNorm(d_last_token_vec, 1, 0, 3);

        std::vector<float> final_ln_out(hidden_dim);
        cudaMemcpy(final_ln_out.data(), d_last_token_vec, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> logits(kVocabSize);
        for (size_t v = 0; v < kVocabSize; ++v){
            float* vocab_row = model_weights.weight_token_emb + (v * kModelSize);
            logits[v] = ops::DotProd(final_ln_out.data(), vocab_row, kModelSize);
        }

        // Argmax
        int best_token_id = 0;
        float max_logit = logits[0];
        for (size_t v = 1; v < kVocabSize; ++v){
            if (logits[v] > max_logit){
                max_logit = logits[v];
                best_token_id = v;
            }
        }

        output_tokens.push_back(best_token_id);
        tokens.push_back(best_token_id);

        std::vector<int> output_vec = {best_token_id};
        std::string pred_output = tokenizer.Decoder(output_vec);
        std::cout << pred_output << std::flush;

        if (best_token_id == 50256){
            std::cout<<"\n[End of text token reached]"<<std::endl;
            break;
        }
        generated_count++;
    } // end while loop

    std::cout<<"output_tokens size: "<< output_tokens.size()<<std::endl;
    std::string final_output = tokenizer.Decoder(output_tokens);
    std::cout<<"OUTPUT: " << final_output<<std::endl;
    /*
     * uncomment if need to double check
     std::cout << "Predicted Token ID: " << best_token_id << std::endl;
     std::vector<int> output_vec = {best_token_id};
     std::string pred_output = tokenizer.Decoder(output_vec);
     std:: cout<< "Test output: "<< pred_output <<std::endl;
     */

    cudaFree(d_tokens);
    cudaFree(d_input_buffer);
    cudaFree(d_ln_buffer);
    cudaFree(d_attention_output);
    cudaFree(d_ff_output);
    cudaFree(d_ff_buffer);
    return 0;
};
