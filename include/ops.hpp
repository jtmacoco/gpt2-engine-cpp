#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif
namespace ops{
#ifdef USE_CUDA
    void MatMul(cublasHandle_t handle, const float* A, const float* B, float* C,
            int M, int N, int K, const float* bias = nullptr);
    void MatMulTransposedB(cublasHandle_t handle, const float* A, const float* B, float* C,int M, int N, int K);
    void AttentionSoftMax(float* d_scores, int num_rows, int row_length, cudaStream_t stream); 
    void Gelu(float* d_buffer, int size, cudaStream_t stream);
    void LayerNorm(float* d_x, const float* d_beta, const float* d_gamma, int dim, cudaStream_t stream);
    void AddResidual(float* d_target, const float* d_source, int size, cudaStream_t stream);
    void ApplyEmbedding(const int* d_tokens, const float* d_token_emb, const float* d_pos_emb, float* d_output, int seq_len, int hidden_dim, int current_pos, cudaStream_t stream);
    void AppendKV(const float* new_K, const float* new_V, float* K_cache, float* V_cache, int current_pos, int max_seq_len, int num_heads, int head_dim, cudaStream_t stream);
    void ScaleScores(float* d_scores, int total_elements, float scale);
    void BulkAppendKV(const float* d_new_K, const float* d_new_V, float* d_K_cache, float* d_V_cache, int seq_len, int max_seq_len, int num_heads, int head_dim, cudaStream_t stream);
    void ArgMax(const float* d_logits, int n, int* d_out_idx, cudaStream_t stream); 

    __global__ void ScaledKernel(float* scores, int total_elements, float scale); 
    __global__ void UnfusedQKVKernel(const float* qkv_buffer, float* Q, float* K, float* V, int seq_len, int num_heads, int head_dim);
    __global__ void ScaledAndMaskKernel(float* score_head, int seq_len, float scale);
    __global__ void AttentionSoftMaxKernel(float* scores, int seq_len);
    __global__ void ConcatHeadsKernel(const float* context_layer, float* proj_output, int seq_len, int num_heads, int head_dim);

#else
    void MatMul(const float* A, const float* B, float* C,
            int M, int N, int K, const float* bias = nullptr);
    void MatMulTransposedB(const float* A, const float* B, float* C,
            int M, int N, int K);
#endif

    void SoftMax(float *x, int size);
    float DotProd(const float* A, const float* B, int length);
}
