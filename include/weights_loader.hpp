#ifndef WEIGHTS_LOADER_HPP
#define WEIGHTS_LOADER_HPP

#include <vector>
#include <string>


//embeddings shape (50257, 768)

namespace WeightsLoader{
    constexpr int kVocabSize = 50257;
    constexpr int kDimensions = 768;
    constexpr size_t kTotalElements = (size_t)kVocabSize*kDimensions;

    std::vector<float> load_weights(const std::string file_path);
}
#endif
