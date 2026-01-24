#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP
#include <iostream>
#include <vector>
#include "weights_loader.hpp"

class InferenceEngine{
    public:
        InferenceEngine(const GPT2Weights& weights);
        void ApplyEmbedding(const std::vector<int>& tokens, float* output_buffer);
        void ApplyLayerNorm(float* x, float* beta, float* gamma, int dim);

    private:
        const GPT2Weights& weights_;
};
#endif
