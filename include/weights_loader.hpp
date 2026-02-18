#ifndef WEIGHTS_LOADER_HPP
#define WEIGHTS_LOADER_HPP

#include <vector>
#include <string>


//embeddings shape (50257, 768)
constexpr int kVocabSize   = 50257;
constexpr int kModelSize   = 768;
constexpr int kMaxSequence = 1024;
constexpr int kFFNSize = 4 * kModelSize; // 768 * 4 = 3072

struct GPT2Weights{
    float* weight_token_emb;
    float* weight_pos_emb;

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
    //calculate the position of the weights
    void map_from_vector(std::vector<float>& full_blob){
        float* ptr = full_blob.data();

        weight_token_emb = ptr;
        ptr += (kVocabSize * kModelSize);

        weight_pos_emb = ptr;
        ptr += (kMaxSequence * kModelSize);

        ln_1_gamma = ptr;
        ptr += kModelSize;

        ln_1_beta = ptr;
        ptr += kModelSize;

        //Attention weights
        qkv_weights = ptr;
        ptr += (kModelSize * 3 * kModelSize);

        qkv_bias = ptr;
        ptr += (3 * kModelSize);

        proj_weights = ptr;
        ptr += (kModelSize * kModelSize);

        proj_bias = ptr;
        ptr += kModelSize;

        ln_2_gamma = ptr;
        ptr += kModelSize;

        ln_2_beta = ptr;
        ptr += kModelSize;

        fc_weights = ptr;
        ptr += (kModelSize + kFFNSize);

        fc_bias = ptr;
        ptr += kFFNSize;

        proj_weights2 = ptr;
        ptr += (kModelSize + kFFNSize);

        proj_bias2 = ptr;
        ptr += kModelSize;

    }
};
namespace WeightsLoader {
    // WTE + WPE
    constexpr size_t kEmbeddings = (size_t)((kVocabSize + kMaxSequence) * kModelSize);

    // LN1 Gamma + Beta
    constexpr size_t kLayerNorm = (2 * kModelSize);

    // QKV Weights (768 * 2304) + Bias (2304)
    constexpr size_t kAttnQKV = (size_t)(kModelSize * 3 * kModelSize) + (3 * kModelSize);

    // Proj Weights (768 * 768) + Bias (768)
    constexpr size_t kAttnProj = (size_t)(kModelSize * kModelSize) + kModelSize;

    //MLP/FFN elements
    constexpr size_t kFFN = (size_t)(kModelSize * kFFNSize) + kFFNSize + (kFFNSize * kModelSize) + kModelSize;

    // The Sum of Everything
    constexpr size_t kTotalElements = kEmbeddings + kLayerNorm + kAttnQKV + kAttnProj + kFFN;

    std::vector<float> load_weights(const std::string file_path);
};
#endif
