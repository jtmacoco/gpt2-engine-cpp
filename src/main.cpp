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

    std::string input = "Hello world";

    std::vector<int> tokens = tokenizer.Encoder(input);
    int seq_len = tokens.size();
    std::vector<float> input_buffer(seq_len * kModelSize);


    InferenceEngine inference_engine(model_weights);
    inference_engine.ApplyEmbedding(tokens,input_buffer.data());
    for (size_t i = 0; i < seq_len; ++i){
        float* current_token_vec = input_buffer.data() +  (i * kModelSize);
        inference_engine.ApplyLayerNorm(current_token_vec,
                model_weights.ln_1_beta,
                model_weights.ln_1_gamma,
                kModelSize);

    }
    std::cout<<"input buffer test ";
    for(int i = 0; i < 5; ++i){
        std::cout<<input_buffer[i]<<" ";
    }
    std::cout<<std::endl;

    int qkv_dim = 3*kModelSize;
    std::vector<float> attention_debug_output(seq_len*qkv_dim);

    inference_engine.AttentionLayer(
            input_buffer.data(),
            attention_debug_output.data(),
            seq_len
            );

    
    std::cout << "Attention Output (First 5 values): ";
    for(int i = 0; i < 5; ++i) {
        std::cout << attention_debug_output[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Attention Output (Last 5 values): ";
    for(int i = 5; i > 0; --i) {
        std::cout << attention_debug_output[attention_debug_output.size()-i-1] << " ";
    }
    std::cout << std::endl;
    return 0;
}
