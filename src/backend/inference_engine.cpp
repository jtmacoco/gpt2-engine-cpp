#include "inference_engine.hpp"
#include "ops.hpp"
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
void InferenceEngine::AttentionLayer(float* input, float* output, int seq_len, int layer_idx){
    float* qkv_w = weights_.layers[layer_idx].qkv_weights;
    float* qkv_b = weights_.layers[layer_idx].qkv_bias;

    int hidden_dim = kModelSize;//786 
    int qkv_dim = 3 * kModelSize;//2304
    int num_heads = 12;
    int head_dim = 64;

    qkv_buffer_.resize(seq_len * qkv_dim);
    proj_output_.resize(seq_len * hidden_dim);

    Q_.resize(num_heads * seq_len * head_dim);
    K_.resize(num_heads * seq_len * head_dim);
    V_.resize(num_heads * seq_len * head_dim);

    //Fused projection (combines Q, K, V into buffer)
    //Layout per token in qkv_buffer: [Query (0-767) | Key (768-1535) | Value (1536-2303)]
    ops::MatMul(input,
            qkv_w,
            qkv_buffer_.data(),
            seq_len,
            qkv_dim,
            hidden_dim,
            qkv_b);
    //Iterate & unfuse data into vectors
    for (size_t s = 0; s < seq_len; ++s){
        for (size_t h = 0; h < num_heads; ++h){
            for(size_t d = 0; d < head_dim; ++d){

                //Source index in qkv_buffer (fused layout so linear)
                //Each token has 3 * hidden_dim values 
                int src_idx = s * (3 * hidden_dim) + (h * head_dim + d);

                //Destination index where it ends
                //[Head, Sequence, Dimension]
                int dst_idx = h * (seq_len * head_dim) + s * head_dim + d;

                Q_[dst_idx] = qkv_buffer_[src_idx];
                K_[dst_idx] = qkv_buffer_[src_idx + hidden_dim];
                V_[dst_idx] = qkv_buffer_[src_idx + (hidden_dim * 2)];
            }//end for d
        }//end for h
    }//end for s
    std::vector<float> scores(num_heads * seq_len * seq_len);
    std::vector<float> context_layer(num_heads * seq_len * head_dim);
    float scale = 1.0f / sqrtf(head_dim);

    for (size_t h = 0; h < num_heads; ++h){
        //Unpack the qkv data 
        float* q_head = Q_.data() + (h * seq_len * head_dim);
        float* k_head = K_.data() + (h * seq_len * head_dim);
        float* v_head = V_.data() + (h * seq_len * head_dim);

        float* score_head = scores.data() + (h* seq_len * seq_len);
        float* context_head = context_layer.data() + (h * seq_len * head_dim);

        ops::MatMulTransposedB(q_head, k_head, score_head, seq_len, seq_len, head_dim);

        for (size_t i = 0; i < seq_len; ++i){
            for (size_t j = 0; j < seq_len; ++j){
                if (j > i){
                    score_head[i * seq_len + j] = -1e9f;
                } else{
                    score_head[i * seq_len + j] *= scale; 
                }
            }//end j loop
            ops::SoftMax(&score_head[i * seq_len], seq_len);
        }// end i loop 
        ops::MatMul(score_head, v_head, context_head, seq_len, head_dim, seq_len, nullptr);
    }// end h loop
    for (size_t s = 0; s < seq_len; ++s){
        for (size_t h = 0; h < num_heads; ++h){
            int src_offset = h * (seq_len * head_dim) + (s * head_dim);
            int dst_offset = s * (num_heads * head_dim) + (h * head_dim);

            float* src_ptr = context_layer.data() + src_offset;
            float* dst_ptr = proj_output_.data() + dst_offset;

            std::copy(src_ptr, src_ptr + head_dim, dst_ptr);
        }//end h loop
    }//end s loop
    ops::MatMul(proj_output_.data(),
                weights_.layers[layer_idx].proj_weights,
                output,
                seq_len,
                hidden_dim,
                hidden_dim,
                weights_.layers[layer_idx].proj_bias);
}

void InferenceEngine::FeedForwardLayer(float* input, float* output, float* buffer, int seq_len, int layer_idx){
    float* fc_w   = weights_.layers[layer_idx].fc_weights;
    float* fc_b   = weights_.layers[layer_idx].fc_bias;
    float* proj_w = weights_.layers[layer_idx].proj_weights2;
    float* proj_b = weights_.layers[layer_idx].proj_bias2;

    int d_ff = kModelSize * 4;// 768 * 4
    //Up projection 
    ops::MatMul(input, fc_w, buffer, seq_len, d_ff, kModelSize, fc_b);
    int buffer_size = seq_len * d_ff;
    //GELU activation
    for(size_t i = 0; i < buffer_size; ++i){
        float x = buffer[i];
        float x_cubed = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
        buffer[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    //Down projection
    ops::MatMul(buffer, proj_w, output, seq_len, kModelSize, d_ff, proj_b);
}

InferenceEngine::~InferenceEngine(){}
