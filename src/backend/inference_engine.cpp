#include "inference_engine.hpp"
#include <cmath>

InferenceEngine::InferenceEngine(const GPT2Weights& weights):weights_(weights){}

void InferenceEngine::ApplyEmbedding(const std::vector<int>& tokens, float* output_buffer){
    int seq_len = tokens.size();
    if (seq_len > kMaxSequence){
        std::cerr << "Error: Input Sequence (" << seq_len << ")execeeds maximum model context" << std::endl;
        return;
    }

    //iterate through tokens to compute embeddings
    for (size_t t = 0; t < seq_len; ++t){
        int token_id = tokens[t];
        int position_id = t;

        //compute where vectors are in memory
        float* token_vec = weights_.weight_token_emb + (token_id * kModelSize);
        float* pos_vec   = weights_.weight_pos_emb   + (position_id * kModelSize);

        //ex t=0 jump 0 steps, t=1 jum 768 steps
        float* target_vec = output_buffer + (t* kModelSize);

        //compute embedding by combining token and position vectors
        for (size_t i = 0; i < kModelSize; ++i){
            target_vec[i] = token_vec[i] + pos_vec[i];
        }
    }
}

void InferenceEngine::ApplyLayerNorm(float* x, float* beta, float* gamma, int dim){
    float sum = 0;
    float sum_square = 0;

    for (size_t i = 0; i < dim; ++i){
        sum+=x[i];
        sum_square+=x[i]*x[i];
    }
    float mean = sum/dim;

    float var = (sum_square/dim) - (mean*mean);

    //prevent divding by 0
    float esp = 1e-5;

    float std_dev = std::sqrt(var+esp);

    //normalize & apply learned gamma and beta
    for (size_t i = 0; i < dim; ++i){
        float norm = (x[i] - mean) / std_dev;//normalization
        x[i] = (norm * gamma[i]) + beta[i];//transformation
    }
}
