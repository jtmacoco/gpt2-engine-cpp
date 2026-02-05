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

    std::string vocab_path  = "data/vocab.json";
    std::string merges_path = "data/merges.txt";

    Tokenizer tokenizer(vocab_path,merges_path);
    
    std::string input = "Hello world";

    std::vector<int> tokens = tokenizer.Encoder(input);
    int seq_len = tokens.size();
    GPT2Weights model_weights;

    model_weights.map_from_vector(weights);
    std::vector<float> output_buffer(seq_len * kModelSize);


    InferenceEngine inference_engine(model_weights);
    inference_engine.ApplyEmbedding(tokens,output_buffer.data());
    for (size_t i = 0; i < seq_len; ++i){
        float* current_token_vec = output_buffer.data() +  (i * kModelSize);
        inference_engine.ApplyLayerNorm(current_token_vec,
                model_weights.ln_1_beta,
                model_weights.ln_1_gamma,
                kModelSize);

    }
    return 0;
}
