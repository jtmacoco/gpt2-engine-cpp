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
    std::vector<float> ln_buffer(seq_len * kModelSize);
    std::vector<float> attention_output(seq_len * kModelSize);

    InferenceEngine inference_engine(model_weights);
    inference_engine.ApplyEmbedding(tokens,input_buffer.data());

    std::copy(input_buffer.begin(), input_buffer.end(), ln_buffer.begin());
    for (size_t i = 0; i < seq_len; ++i){
        float* current_token_vec = ln_buffer.data() + (i * kModelSize);
        inference_engine.ApplyLayerNorm(current_token_vec,
                model_weights.ln_1_beta,
                model_weights.ln_1_gamma,
                kModelSize);

    }
    std::cout << "--- LN Output (First 5) ---" << std::endl;
    for(int i=0; i<5; ++i) std::cout << ln_buffer[i] << " ";
    std::cout << std::endl;
    inference_engine.AttentionLayer(
            ln_buffer.data(),
            attention_output.data(),
            seq_len
            );
    std::cout << "--- Attention Output (First 10 values) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6); // Show 6 decimal places
    for (int i = 0; i < 10; ++i) {
        std::cout << attention_output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

