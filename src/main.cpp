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

    //Print tokens
    for (int i = 0; i < tokens.size(); i++){
        std::cout<< tokens[i] << " "; 
    }

    std::cout<<std::endl;

    //Testing decoder
    std::string text = tokenizer.Decoder(tokens);
    std::cout<< text << std::endl;

    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    std::cout << "\n[DEBUG] Verifying Pointer Jumps:" << std::endl;

    // Check Distance: Position Embedding -> Layer Norm Gamma
    // We expect this to be (1024 * 768) = 786,432 floats
    long dist_wpe_to_gamma = model_weights.ln_1_gamma - model_weights.weight_pos_emb;
    std::cout << "Distance (WPE -> Gamma): " << dist_wpe_to_gamma
              << " (Expected: " << (1024 * 768) << ")" << std::endl;

    // Check Distance: Gamma -> Beta
    // We expect this to be exactly 768 floats
    long dist_gamma_to_beta = model_weights.ln_1_beta - model_weights.ln_1_gamma;
    std::cout << "Distance (Gamma -> Beta): " << dist_gamma_to_beta
              << " (Expected: 768)" << std::endl;

    // Check Values
    // Layer Norm Gamma (Scale) usually acts like a multiplier, so values are often near 1.0 or 0.5
    // Layer Norm Beta (Shift) acts like addition, so values are often near 0.0
    std::cout << "Gamma[0]: " << *model_weights.ln_1_gamma << std::endl;
    std::cout << "Beta[0]:  " << *model_weights.ln_1_beta  << std::endl;

    //Check Inference Embedding
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
