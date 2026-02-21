#ifndef WEIGHTS_LOADER_HPP
#define WEIGHTS_LOADER_HPP
#include <vector>
#include <string>

//embeddings shape (50257, 768) 
constexpr int kVocabSize   = 50257;
constexpr int kModelSize   = 768;
constexpr int kMaxSequence = 1024;
constexpr int kFFNSize = 4 * kModelSize; // 768 * 4 = 3072
constexpr int kNumLayers = 12;
struct GPT2LayerWeights{
    float* ln_1_gamma; 
    float* ln_1_beta; 

    float* qkv_weights;
    float* qkv_bias;

    float* proj_weights;
    float* proj_bias;

    float* ln_2_gamma; 
    float* ln_2_beta; 

    float* fc_weights;
    float* fc_bias;

    float* proj_weights2;
    float* proj_bias2;
};
struct GPT2Weights{
    float* weight_token_emb;
    float* weight_pos_emb;
    
    float* ln_f_gamma;
    float* ln_f_beta;
    std::vector<GPT2LayerWeights> layers;

    //calculate the position of the weights
    void map_from_vector(std::vector<float>& full_blob){
        float* ptr = full_blob.data();

        weight_token_emb = ptr;
        ptr += (kVocabSize * kModelSize);

        weight_pos_emb = ptr;
        ptr += (kMaxSequence * kModelSize);
        layers.resize(kNumLayers);
        for (size_t i = 0; i < kNumLayers; ++i){
            layers[i].ln_1_gamma    = ptr; ptr += kModelSize;
            layers[i].ln_1_beta     = ptr; ptr += kModelSize;

            layers[i].qkv_weights   = ptr; ptr += (kModelSize * 3 * kModelSize);
            layers[i].qkv_bias      = ptr; ptr += (3 * kModelSize);

            layers[i].proj_weights  = ptr; ptr += (kModelSize * kModelSize);
            layers[i].proj_bias     = ptr; ptr += kModelSize;

            layers[i].ln_2_gamma    = ptr; ptr += kModelSize;
            layers[i].ln_2_beta     = ptr; ptr += kModelSize;

            layers[i].fc_weights    = ptr; ptr += (kModelSize * kFFNSize);
            layers[i].fc_bias       = ptr; ptr += kFFNSize;

            layers[i].proj_weights2 = ptr; ptr += (kFFNSize * kModelSize);
            layers[i].proj_bias2    = ptr; ptr += kModelSize;
        }// end i loop
        ln_f_gamma = ptr; ptr += kModelSize;
        ln_f_beta = ptr; ptr += kModelSize;
    }
};
namespace WeightsLoader {
    constexpr size_t kEmbeddings = (size_t)((kVocabSize + kMaxSequence) * kModelSize);
    constexpr size_t kLayerNorm  = (2 * kModelSize);
    constexpr size_t kAttnQKV    = (size_t)(kModelSize * 3 * kModelSize) + (3 * kModelSize);
    constexpr size_t kAttnProj   = (size_t)(kModelSize * kModelSize) + kModelSize;
    constexpr size_t kFFN        = (size_t)(kModelSize * kFFNSize) + kFFNSize + (kFFNSize * kModelSize) + kModelSize;

    constexpr size_t kLayerElements = kLayerNorm * 2 + kAttnQKV + kAttnProj + kFFN;
    constexpr size_t kTotalElements = kEmbeddings + (kLayerElements * kNumLayers) + kLayerNorm;
    std::vector<float> load_weights(const std::string file_path);
};
#endif
