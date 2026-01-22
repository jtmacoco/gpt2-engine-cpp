#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
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
    std::vector<float> output_buffer(seq_len * kModelSize);

    //print tokens
    for (int i = 0; i < tokens.size(); i++){
        std::cout<< tokens[i] << " "; 
    }

    std::cout<<std::endl;

    //testing decoder
    std::string text = tokenizer.Decoder(tokens);
    std::cout<< text << std::endl;

    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    InferenceEngine inference_engine(model_weights);
    std::cout << "Running ApplyEmbedding..." << std::endl;
    inference_engine.ApplyEmbedding(tokens,output_buffer.data());
    std::cout << " \n[verification] First 3 floats of Token 0:" << std::endl;
    for (size_t i = 0; i < 3; ++i){
        std::cout << std::fixed << std::setprecision(6) << output_buffer[i] << " ";
    }
    std::cout<<std::endl;
    return 0;
}
