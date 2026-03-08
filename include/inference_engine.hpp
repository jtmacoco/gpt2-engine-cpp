#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP
#include <iostream> 
#include <vector> 
#include "weights_loader.hpp"

#ifdef USE_CUDA
#include <cublas_v2.h>
#endif

class InferenceEngine{
    public:
        InferenceEngine(const GPT2Weights& weights);
        void FeedForwardLayer(float* input, float* output, float* buffer, int seq_len, int layer_idx);

#ifdef USE_CUDA
        void ApplyLayerNorm(float* d_x, int seq_len, int layer_idx, int ln_type);
        void AttentionLayer(float* input,float* output, int seq_len, int layer_idx, int current_pos);
        void ApplyEmbedding(const int* d_tokens, int seq_len, float* d_output_buffer, int current_pos);
        int SampleNextToken(const float* d_last_hidden); 

        cudaStream_t GetStream() const {return stream_;};
        cublasHandle_t GetCublasHandle() const {return cublas_handle_;};
#else
        void AttentionLayer(float* input,float* output, int seq_len, int layer_idx);
        void ApplyEmbedding(const std::vector<int>& tokens, float* output_buffer);
        void ApplyLayerNorm(float* x, float* beta, float* gamma, int dim);
#endif
        ~InferenceEngine();


    private:
        std::vector<float> qkv_buffer_;
        std::vector<float> proj_output_;
        std::vector<float> Q_, K_, V_;
        const GPT2Weights& weights_;

#ifdef USE_CUDA
        float* d_qkv_buffer_  = nullptr;
        float* d_proj_output_ = nullptr;

        float* d_Q_ = nullptr;
        float* d_K_ = nullptr;
        float* d_V_ = nullptr;

        float* d_K_cache_ = nullptr;
        float* d_V_cache_ = nullptr;

        float* d_scores_        = nullptr;
        float* d_context_layer_ = nullptr;


        float* d_token_emb_ = nullptr;
        float* d_logits_    = nullptr;
        int* d_best_token_  = nullptr;

        GPT2DeviceWeights d_weights_;
        cublasHandle_t cublas_handle_;
        cudaStream_t stream_;

#endif
};
#endif
