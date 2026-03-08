#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
#include "ops.hpp"
#include <iostream>
#include <filesystem>
#include <string>
namespace fs = std::filesystem;


int main(int argc, char** argv){
    //Load weights 
    std::string weights_file = "data/gpt2_embeddings.bin";

    //contains both positional and token weights
    auto weights = WeightsLoader::load_weights(weights_file);
    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    //Handle BPE Tokenizer
    std::string vocab_path  = "data/vocab.json";
    std::string merges_path = "data/merges.txt";
    Tokenizer tokenizer(vocab_path,merges_path);

    //Basic input used change later
    std::string input = "The quick brown fox jumps over the lazy";
    int max_tokens_to_generate = 20;
    int generated_count = 0;

    std::vector<int> tokens = tokenizer.Encoder(input);
    std::cout<< "Input: " << input <<std::endl;
    std::vector<int> output_tokens;

    int max_seq_len = kMaxSequence;
    int hidden_dim = kModelSize;

    //Buffer allocations
    int* d_tokens = nullptr;
    cudaMalloc((void**)&d_tokens, max_seq_len * sizeof(int));

    float* d_input_buffer     = nullptr;//Current hidden state
    float* d_ln_buffer        = nullptr;//Buffer for LayerNrom
    float* d_attention_output = nullptr;//Output of MHA
    float* d_ff_output        = nullptr;//Output of MLP
    float* d_ff_buffer        = nullptr;//Middle layers of MLP (4x hidden_dim)

    cudaMalloc((void**)&d_input_buffer,    max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_ln_buffer,        max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_attention_output, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_ff_output,        max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_ff_buffer,        max_seq_len * hidden_dim * 4 * sizeof(float));

    //Init engine
    InferenceEngine inference_engine(model_weights);
    cudaStream_t stream = inference_engine.GetStream();
    std::vector<int> current_input_tokens = tokens;
    int current_pos = 0;
    while (generated_count < max_tokens_to_generate) {
        int seq_len = current_input_tokens.size();
        
        //Copy tokens to GPU
        cudaMemcpyAsync(d_tokens,
                current_input_tokens.data(),
                seq_len * sizeof(int),
                cudaMemcpyHostToDevice,
                stream);

        //Convert tokens to vectors
        inference_engine.ApplyEmbedding(d_tokens, seq_len, d_input_buffer, current_pos);

        int buffer_bytes = seq_len * kModelSize * sizeof(float);

        //Transformer Layers
        for (size_t layer_idx = 0; layer_idx < kNumLayers; ++layer_idx){
            //Atention
            cudaMemcpyAsync(d_ln_buffer,
                d_input_buffer,
                buffer_bytes,
                cudaMemcpyDeviceToDevice,
                stream);
            inference_engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 1);
            inference_engine.AttentionLayer(d_ln_buffer, d_attention_output, seq_len, layer_idx, current_pos);
            ops::AddResidual(d_input_buffer, d_attention_output, seq_len * kModelSize, stream);

            //FF (MLP)
            cudaMemcpyAsync(d_ln_buffer, d_input_buffer, buffer_bytes, cudaMemcpyDeviceToDevice, stream);
            inference_engine.ApplyLayerNorm(d_ln_buffer, seq_len, layer_idx, 2);
            inference_engine.FeedForwardLayer(d_ln_buffer, d_ff_output, d_ff_buffer, seq_len, layer_idx);

            ops::AddResidual(d_input_buffer, d_ff_output, seq_len * kModelSize, stream);
        }

        // The math naturally handles both phases.
        // If seq_len == 1 (decode), this points to index 0.
        // If seq_len == N (prefill), this points to the last token of the prompt.
        float* d_last_token_vec = d_input_buffer + ((seq_len - 1) * hidden_dim);

        //Final Normalization
        inference_engine.ApplyLayerNorm(d_last_token_vec, 1, 0, 3);

        //Argmax
        int best_token_id = inference_engine.SampleNextToken(d_last_token_vec);
        output_tokens.push_back(best_token_id);

        std::vector<int> output_vec = {best_token_id};

        std::string pred_output = tokenizer.Decoder(output_vec);
        std::cout << pred_output << std::flush;

        if (best_token_id == 50256){
            std::cout<<"\n[End of text token reached]"<<std::endl;
            break;
        }

        //update state for next iter
        current_pos += seq_len;
        current_input_tokens = {best_token_id};
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

    //Cleanup
    cudaFree(d_tokens);
    cudaFree(d_input_buffer);
    cudaFree(d_ln_buffer);
    cudaFree(d_attention_output);
    cudaFree(d_ff_output);
    cudaFree(d_ff_buffer);
    return 0;
};
