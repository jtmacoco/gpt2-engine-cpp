#include "inference_engine.hpp"
#include "ops.hpp"
#include <cmath>
InferenceEngine::InferenceEngine(const GPT2Weights& weights):weights_(weights){
    cublasStatus_t stat = cublasCreate(&cublas_handle_);
    if (stat != CUBLAS_STATUS_SUCCESS)
        std::cerr << " Failed to init cuBLAS in Inference Engine" << std::endl;
    int max_seq_len = 1024;//TODO Adjust
    int hidden_dim = kModelSize;
    int qkv_dim = 3 * kModelSize;
    int num_heads = 12;
    int head_dim = 64;

    float* d_raw_model_blob = nullptr;
    size_t total_bytes = WeightsLoader::kTotalElements * sizeof(float);
    cudaMalloc((void**)&d_raw_model_blob, total_bytes);
    cudaMemcpy(d_raw_model_blob, weights.weight_token_emb, total_bytes, cudaMemcpyHostToDevice);

    d_weights_.map_from_device_pointer(d_raw_model_blob);

    cudaMalloc((void**)&d_qkv_buffer_, max_seq_len * qkv_dim     * sizeof(float));
    cudaMalloc((void**)&d_proj_output_, max_seq_len * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_Q_, num_heads * max_seq_len * head_dim * sizeof(float));
    cudaMalloc((void**)&d_K_, num_heads * max_seq_len * head_dim * sizeof(float));
    cudaMalloc((void**)&d_V_, num_heads * max_seq_len * head_dim * sizeof(float));
    cudaMalloc((void**)&d_scores_, num_heads * max_seq_len * max_seq_len * sizeof(float));
    cudaMalloc((void**)&d_context_layer_, num_heads * max_seq_len * head_dim * sizeof(float));
}

void InferenceEngine::ApplyEmbedding(const int* d_tokens, int seq_len, float* d_output_buffer) {
    ops::ApplyEmbedding(
        d_tokens, 
        d_weights_.weight_token_emb, 
        d_weights_.weight_pos_emb, 
        d_output_buffer, 
        seq_len, 
        kModelSize
    );
    cudaDeviceSynchronize();
}

void InferenceEngine::ApplyLayerNorm(float* d_x, int seq_len, int layer_idx, int ln_type) {
    float* d_beta = nullptr;
    float* d_gamma = nullptr;

    // 1. Select the correct GPU weights based on the type
    if (ln_type == 1) {
        d_beta = d_weights_.layers[layer_idx].ln_1_beta;
        d_gamma = d_weights_.layers[layer_idx].ln_1_gamma;
    } else if (ln_type == 2) {
        d_beta = d_weights_.layers[layer_idx].ln_2_beta;
        d_gamma = d_weights_.layers[layer_idx].ln_2_gamma;
    } else if (ln_type == 3) {
        // We will use type 3 for the final LayerNorm before the Logits!
        d_beta = d_weights_.ln_f_beta;
        d_gamma = d_weights_.ln_f_gamma;
    }

    int dim = kModelSize; // 768

    // 2. Loop through every token in the sequence and normalize it
    for (int i = 0; i < seq_len; ++i) {
        // Calculate the pointer offset for this specific token
        float* d_current_vec = d_x + (i * dim);

        // Launch your custom LayerNorm kernel!
        ops::LayerNorm(d_current_vec, d_beta, d_gamma, dim);
    }

    // 3. Wait for all tokens to finish normalizing
    cudaDeviceSynchronize();
}
void InferenceEngine::AttentionLayer(float* input, float* output, int seq_len, int layer_idx){
    float* d_qkv_w = d_weights_.layers[layer_idx].qkv_weights;
    float* d_qkv_b = d_weights_.layers[layer_idx].qkv_bias;

    int hidden_dim = kModelSize;//786 
    int qkv_dim = 3 * kModelSize;//2304
    int num_heads = 12;
    int head_dim = 64;
    int threads = 256;

    //Fused projection (combines Q, K, V into buffer)
    //Layout per token in qkv_buffer: [Query (0-767) | Key (768-1535) | Value (1536-2303)]
    ops::MatMul(
            cublas_handle_,
            input,
            d_qkv_w,
            d_qkv_buffer_,
            seq_len,
            qkv_dim,
            hidden_dim,
            d_qkv_b);

    //Iterate & unfuse data into vectors
    int total_elements = seq_len * num_heads * head_dim;
    int blocks = (total_elements + threads - 1) / threads;
    ops::UnfusedQKVKernel<<<blocks, threads>>>(
            d_qkv_buffer_, d_Q_, d_K_, d_V_,
            seq_len, num_heads, head_dim
            );

    cudaDeviceSynchronize();

    float scale = 1.0f / sqrtf(head_dim);

    for (size_t h = 0; h < num_heads; ++h){
        //Unpack the qkv data 
        float* d_q_head = d_Q_ + (h * seq_len * head_dim);
        float* d_k_head = d_K_ + (h * seq_len * head_dim);
        float* d_v_head = d_V_ + (h * seq_len * head_dim);

        float* d_score_head = d_scores_ + (h * seq_len * seq_len);//TODO fix
        float* d_context_head = d_context_layer_ + (h * seq_len * head_dim);

        ops::MatMulTransposedB(cublas_handle_, d_q_head, d_k_head, d_score_head, seq_len, seq_len, head_dim);

        int total_scores = seq_len * seq_len;
        int threads = 256;
        int blocks = (total_scores + threads - 1) / threads;
        ops::ScaledAndMaskKernel<<<blocks, threads>>>(d_score_head, seq_len, scale);
        cudaDeviceSynchronize();
        ops::AttentionSoftMax(d_score_head, seq_len);
        ops::MatMul(cublas_handle_, d_score_head, d_v_head, d_context_head, seq_len, head_dim, seq_len, nullptr);
    }// end h loop
    cudaDeviceSynchronize();
    int total_concat_elements = seq_len * num_heads * head_dim;
    int concat_threads = 256;
    int concat_blocks = (total_concat_elements + concat_threads - 1) / concat_threads;

    ops::ConcatHeadsKernel<<<concat_blocks, concat_threads>>>(
            d_context_layer_,
            d_proj_output_,
            seq_len, num_heads, head_dim
    );
    cudaDeviceSynchronize();

    float* d_proj_w = d_weights_.layers[layer_idx].proj_weights;
    float* d_proj_b = d_weights_.layers[layer_idx].proj_bias;
     
    ops::MatMul(
        cublas_handle_,
        d_proj_output_,
        d_proj_w,
        output,         
        seq_len,
        hidden_dim,
        hidden_dim,
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
    ops::Gelu(buffer, buffer_size);

    cudaDeviceSynchronize();

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
}
