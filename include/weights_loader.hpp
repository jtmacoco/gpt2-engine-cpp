#ifndef WEIGHTS_LOADER_HPP
#define WEIGHTS_LOADER_HPP

#include <vector>
#include <string>


//embeddings shape (50257, 768)
constexpr int kVocabSize   = 50257;
constexpr int kModelSize   = 768;
constexpr int kMaxSequence = 1024;

struct GPT2Weights{
    float* weight_token_emb;
    float* weight_pos_emb;

    void map_from_vector(std::vector<float>& full_blob){
        weight_token_emb = full_blob.data();
        weight_pos_emb = weight_token_emb + (kVocabSize * kModelSize);
    }
};
namespace WeightsLoader{
    constexpr size_t kTotalElements = (size_t)((kVocabSize + kMaxSequence) * kModelSize);

    std::vector<float> load_weights(const std::string file_path);
};
#endif
