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
    float A[] = {1.0f, 2.0f, 3.0f};
    float B[] = {4.0f, 7.0f, 5.0f, 8.0f, 6.0f, 9.0f};
    int M = 1;
    int K = 3;
    int N = 2;
    float* C = new float[M*N];
    ops::MatMul(A, B, C, M, N, K);

    for (size_t i = 0; i < K; i++){
        std::cout << C[i]<< " ";
    }
    std::cout<<std::endl;

    //void MatMul(const float* A, const float* B, float* C, int M, int N, int K, const float* bias = nullptr)

    std::vector<float> input_test = {1.0f,2.0f,3.0f};
    ops::SoftMax(input_test.data(), input_test.size());
    std::cout << "Test 1 (Standard): ";
    float sum = 0.0f;
    for (float v : input_test) {
        std::cout << v << " ";
        sum += v;
    }
    std::cout << "| Sum: " << sum << std::endl;
    return 0;
}
