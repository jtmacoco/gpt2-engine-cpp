#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP
#include <iostream>
#include "weights_loader.hpp"

class InferenceEngine{
    public:
        InferenceEngine(const GPT2Weights& weights);

    private:
        const GPT2Weights& weights_;
        void ApplyEmbedding(const std::vector<int>& tokens, float* output_buffer);
};
#endif
