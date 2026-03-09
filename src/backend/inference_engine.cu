#include "inference_engine.hpp"
#include "ops.hpp"
#include <cmath>
InferenceEngine::InferenceEngine(const GPT2Weights& weights):weights_(weights){
    //Init CUDA libraries
    cublasStatus_t stat = cublasCreate(&cublas_handle_);
    if (stat != CUBLAS_STATUS_SUCCESS)
        std::cerr << " Failed to init cuBLAS in Inference Engine" << std::endl;
    //MODEL HYPERPARAMETERS
    int max_seq_len = 1024;           //Max context window maybe adjust
    int qkv_dim     = 3 * kModelSize; //combined dimensions for Query, Key, Value projects
    int num_heads   = 12;             //Num of attention heads
    int head_dim    = 64;             //Dimension per head (12 * 64 = 768)
    int num_layers  = 12;             //Total Transformer blocks



    //Stream setup
    cudaStreamCreate(&stream_);
    cublasSetStream(cublas_handle_, stream_);

    //Model weight management 
    float* d_raw_model_blob = nullptr;
    size_t total_bytes = WeightsLoader::kTotalElements * sizeof(float);

    //Allocate block for all model weights to improve cache
    cudaMalloc((void**)&d_raw_model_blob, total_bytes);
    cudaMemcpy(d_raw_model_blob, weights.weight_token_emb, total_bytes, cudaMemcpyHostToDevice);

    //allocate weight structures to GPU basically
    d_weights_.map_from_device_pointer(d_raw_model_blob);

    //Buffer allocation
    cudaMalloc((void**)&d_qkv_buffer_,    max_seq_len * qkv_dim     * sizeof(float));
    cudaMalloc((void**)&d_proj_output_,   max_seq_len * kModelSize * sizeof(float));
    cudaMalloc((void**)&d_Q_,             num_heads * max_seq_len * head_dim * sizeof(float));
    cudaMalloc((void**)&d_K_,             num_heads * max_seq_len * head_dim * sizeof(float));
    cudaMalloc((void**)&d_V_,             num_heads * max_seq_len * head_dim * sizeof(float));
    cudaMalloc((void**)&d_scores_,        num_heads * max_seq_len * max_seq_len * sizeof(float));
    cudaMalloc((void**)&d_context_layer_, num_heads * max_seq_len * head_dim * sizeof(float));

    //KV Cache allocation
    size_t cache_size_per_layer = num_heads * max_seq_len * head_dim;
    cudaMalloc((void**)&d_K_cache_, num_layers * cache_size_per_layer * sizeof(float));
    cudaMalloc((void**)&d_V_cache_, num_layers * cache_size_per_layer * sizeof(float));

    //Allocate and copy embeddings for input stage
    cudaMalloc(&d_token_emb_, kVocabSize * kModelSize * sizeof(float));
    cudaMemcpyAsync(d_token_emb_,
            weights.weight_token_emb,
            kVocabSize * kModelSize * sizeof(float),
            cudaMemcpyHostToDevice,
            stream_);

    //Buffer for final logit calculations and Argmax
    cudaMalloc(&d_logits_, kVocabSize * sizeof(float));
    cudaMalloc(&d_best_token_, sizeof(int));

    cudaMemsetAsync(d_K_cache_, 0, num_layers * cache_size_per_layer * sizeof(float), stream_);
    cudaMemsetAsync(d_V_cache_, 0, num_layers * cache_size_per_layer * sizeof(float), stream_);

}

void InferenceEngine::ApplyEmbedding(const int* d_tokens, int seq_len, float* d_output_buffer, int current_pos) {
    ops::ApplyEmbedding(
            d_tokens, d_weights_.weight_token_emb, d_weights_.weight_pos_emb, 
            d_output_buffer, seq_len, kModelSize, current_pos, stream_
            );
}

void InferenceEngine::ApplyLayerNorm(float* d_x, int seq_len, int layer_idx, int ln_type) {
    float* d_beta = nullptr;
    float* d_gamma = nullptr;

    //Weight Selection Logic
    //GPT2 normalizes before sub layers like attention and mlp
    if (ln_type == 1) {
        d_beta = d_weights_.layers[layer_idx].ln_1_beta;
        d_gamma = d_weights_.layers[layer_idx].ln_1_gamma;
    } else if (ln_type == 2) {
        d_beta = d_weights_.layers[layer_idx].ln_2_beta;
        d_gamma = d_weights_.layers[layer_idx].ln_2_gamma;
    } else if (ln_type == 3) {
        d_beta = d_weights_.ln_f_beta;
        d_gamma = d_weights_.ln_f_gamma;
    }

    //LayerNorm calculated across hidden dim for each token indepedently 
    int dim = kModelSize; // 768

    for (size_t i = 0; i < seq_len; ++i) {
        //Calculate poter offsets
        float* d_current_vec = d_x + (i * dim);

        ops::LayerNorm(d_current_vec, d_beta, d_gamma, dim, stream_);
    }

}
void InferenceEngine::AttentionLayer(float* input, float* output, int seq_len, int layer_idx, int current_pos){
    //load weights
    float* d_qkv_w = d_weights_.layers[layer_idx].qkv_weights;
    float* d_qkv_b = d_weights_.layers[layer_idx].qkv_bias;

    //Constants Change this later
    int qkv_dim = 3 * kModelSize;//2304
    int num_heads = 12;
    int head_dim = 64;
    int threads = 256;

    int max_seq_len = 1024;//TODO add to global 
                           //Cache data
                           //is_prefill is true if we are processing prompt so multiple tokens at once
                           //cache_len is total history (past tokens + current toekns)
    int is_prefill = (seq_len > 1);
    int cache_len = is_prefill ? seq_len : (current_pos + 1);

    //Ofset global cache for specific layer
    size_t layer_offset = layer_idx * (num_heads * max_seq_len * head_dim);
    float* d_layer_k_cache = d_K_cache_ + layer_offset;
    float* d_layer_v_cache = d_V_cache_ + layer_offset;


    //Fused projection (combines Q, K, V into buffer)
    //Layout per token in qkv_buffer: [Query (0-767) | Key (768-1535) | Value (1536-2303)]
    ops::MatMul(
            cublas_handle_,
            input,
            d_qkv_w,
            d_qkv_buffer_,
            seq_len,
            qkv_dim,
            kModelSize,
            d_qkv_b);

    //Reorganize fused buffer into separate  buffers for processing
    int total_elements = seq_len * num_heads * head_dim;
    int blocks = (total_elements + threads - 1) / threads;
    ops::UnfusedQKVKernel<<<blocks, threads, 0, stream_>>>(
            d_qkv_buffer_, d_Q_, d_K_, d_V_,
            seq_len, num_heads, head_dim
            );
    //Update KV cache
    //Store updated K and V vectors into the cache for future tokens
    if (is_prefill) {
        ops::BulkAppendKV(d_K_, d_V_, d_layer_k_cache, d_layer_v_cache, seq_len, max_seq_len, num_heads, head_dim, stream_);
    } else {
        ops::AppendKV(d_K_, d_V_, d_layer_k_cache, d_layer_v_cache, current_pos, max_seq_len, num_heads, head_dim, stream_);
    }


    //Multi-head Attention
    float scale = 1.0f / sqrtf(head_dim);

    for (size_t h = 0; h < num_heads; ++h){
        //Unpack the qkv data 
        float* d_q_head = d_Q_ + (h * seq_len * head_dim);

        //start at head's block 
        float* d_k_head = d_layer_k_cache + (h * max_seq_len * head_dim);
        float* d_v_head = d_layer_v_cache + (h * max_seq_len * head_dim);

        //

        //float* d_score_head = d_scores_ + (h * seq_len * seq_len);//TODO fix
        float* d_score_head = d_scores_ + (h * max_seq_len * max_seq_len);
        float* d_context_head = d_context_layer_ + (h * seq_len * head_dim);
        if (is_prefill){
            //Casual Attention Q * K^T
            ops::MatMulTransposedB(cublas_handle_, d_q_head, d_k_head, d_score_head, seq_len, seq_len, head_dim);
            //Apply scaling and casual mask (remember mask prevent looking at future tokens in Notes NLP!)
            int total_scores = seq_len * seq_len;
            int score_blocks = (total_scores + threads -1) / threads;
            ops::ScaledAndMaskKernel<<<score_blocks, threads, 0, stream_>>>(d_score_head, seq_len, scale);
            //Softmax Across the score matrix
            ops::AttentionSoftMax(d_score_head, seq_len, seq_len, stream_);

            //Compute Context
            ops::MatMul(cublas_handle_, d_score_head, d_v_head, d_context_head, seq_len, head_dim, seq_len, nullptr);
        }else{
            //Incremental Attention 
            ops::MatMulTransposedB(cublas_handle_, d_q_head, d_k_head, d_score_head, 1, cache_len, head_dim);
            //Single row scaling
            int score_blocks = (cache_len + threads - 1) / threads;
            ops::ScaledKernel<<<score_blocks, threads, 0, stream_>>>(d_score_head, cache_len, scale);

            ops::AttentionSoftMax(d_score_head, 1, cache_len, stream_);
            ops::MatMul(cublas_handle_, d_score_head, d_v_head, d_context_head, 1, head_dim, cache_len, nullptr);
        }
    }// end h loop
     //Final Projections
    int total_concat_elements = seq_len * num_heads * head_dim;
    int concat_threads = 256;
    int concat_blocks = (total_concat_elements + concat_threads - 1) / concat_threads;

    ops::ConcatHeadsKernel<<<concat_blocks, concat_threads, 0, stream_>>>(
            d_context_layer_,
            d_proj_output_,
            seq_len, num_heads, head_dim
            );

    float* d_proj_w = d_weights_.layers[layer_idx].proj_weights;
    float* d_proj_b = d_weights_.layers[layer_idx].proj_bias;

    ops::MatMul(
            cublas_handle_,
            d_proj_output_,
            d_proj_w,
            output,         
            seq_len,
            kModelSize,
            kModelSize,
            d_proj_b
            );
}

void InferenceEngine::FeedForwardLayer(float* input, float* output, float* buffer, int seq_len, int layer_idx){
    float* d_fc_w   = d_weights_.layers[layer_idx].fc_weights;
    float* d_fc_b   = d_weights_.layers[layer_idx].fc_bias;
    float* d_proj_w = d_weights_.layers[layer_idx].proj_weights2;
    float* d_proj_b = d_weights_.layers[layer_idx].proj_bias2;

    int d_ff = kModelSize * 4;// 768 * 4
                              //Up projection 
    int buffer_size = seq_len * d_ff;
    ops::MatMul(
            cublas_handle_, 
            input, 
            d_fc_w, 
            buffer, 
            seq_len, 
            d_ff, 
            kModelSize, 
            d_fc_b
            );
    ops::Gelu(buffer, buffer_size, stream_);


    ops::MatMul(
            cublas_handle_,
            buffer,
            d_proj_w,
            output,
            seq_len,
            kModelSize,
            d_ff,
            d_proj_b
            );
}
//Predicts the next token by projecting the final hidden state to vocabulary space.
int InferenceEngine::SampleNextToken(const float* d_last_hidden /*[hidden]*/) {
    ops::MatMulTransposedB(
            cublas_handle_,
            d_last_hidden,
            d_token_emb_,
            d_logits_,
            1, kVocabSize, kModelSize
            );

    //Gready selection
    //Change later and use temperature
    ops::ArgMax(d_logits_, kVocabSize, d_best_token_, stream_);

    int h_best = 0;
    cudaMemcpyAsync(&h_best, d_best_token_, sizeof(int), cudaMemcpyDeviceToHost, stream_);
    //Blocking as must wait for GPU to finish ArgMax and the copy
    cudaStreamSynchronize(stream_); 
    return h_best;
}

void InferenceEngine::ResetKV() {
    const int num_heads   = 12;
    const int head_dim    = 64;
    const int num_layers  = 12;

    size_t cache_elems_per_layer = (size_t)num_heads * (size_t)kMaxSequence * (size_t)head_dim;
    size_t total_cache_elems     = (size_t)num_layers * cache_elems_per_layer;
    size_t total_cache_bytes     = total_cache_elems * sizeof(float);

    // Clear caches on the engine stream (async)
    cudaMemsetAsync(d_K_cache_, 0, total_cache_bytes, stream_);
    cudaMemsetAsync(d_V_cache_, 0, total_cache_bytes, stream_);
}

InferenceEngine::~InferenceEngine(){
    cublasDestroy(cublas_handle_);
    if (d_weights_.weight_token_emb != nullptr) {
        cudaFree(d_weights_.weight_token_emb);
    }
    cudaFree(d_qkv_buffer_);
    cudaFree(d_proj_output_);
    cudaFree(d_Q_);
    cudaFree(d_K_);
    cudaFree(d_V_);
    cudaFree(d_scores_);
    cudaFree(d_context_layer_);
    cudaStreamDestroy(stream_);
    cudaFree(d_K_cache_); // Add this
    cudaFree(d_V_cache_); // Add this
    if (d_token_emb_) cudaFree(d_token_emb_);
    if (d_logits_) cudaFree(d_logits_);
    if (d_best_token_) cudaFree(d_best_token_);
}
