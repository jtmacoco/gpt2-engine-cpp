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

        void AttentionLayer(float* input,float* output, int seq_len, int layer_idx);
        void FeedForwardLayer(float* input, float* output, float* buffer, int seq_len, int layer_idx);

    private:
        std::vector<float> qkv_buffer_;
        std::vector<float> proj_output_;
        std::vector<float> Q_, K_, V_;
        const GPT2Weights& weights_;
};
#endif
