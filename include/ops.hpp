#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif
namespace ops{
#ifdef USE_CUDA
    void MatMul(cublasHandle_t handle, const float* A, const float* B, float* C,
            int M, int N, int K, const float* bias = nullptr);
    void MatMulTransposedB(cublasHandle_t handle, const float* A, const float* B, float* C,
            int M, int N, int K);
    void AttentionSoftMax(float* d_scores, int seq_len);
    void Gelu(float* d_buffer, int size);
    void LayerNorm(float* d_x, const float* d_beta, const float* d_gamma, int dim);
    void AddResidual(float* d_target, const float* d_source, int size);
        void ApplyEmbedding(const int* d_tokens, const float* d_token_emb, const float* d_pos_emb, float* d_output,
        int seq_len, int hidden_dim); 


    __global__ void UnfusedQKVKernel(const float* qkv_buffer, float* Q, float* K, float* V, int seq_len, 
            int num_heads, int head_dim);
    __global__ void ScaledAndMaskKernel(float* score_head, int seq_len, float scale);
    __global__ void AttentionSoftMaxKernel(float* scores, int seq_len);
    __global__ void ConcatHeadsKernel(const float* context_layer, float* proj_output,
                                      int seq_len, int num_heads, int head_dim);

#else
    void MatMul(const float* A, const float* B, float* C,
            int M, int N, int K, const float* bias = nullptr);
    void MatMulTransposedB(const float* A, const float* B, float* C,
            int M, int N, int K);
    #endif

    void SoftMax(float *x, int size);
    float DotProd(const float* A, const float* B, int length);
}
