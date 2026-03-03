#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
namespace ops{
#ifdef USE_CUDA
    void MatMul(cublasHandle_t handle, const float* A, const float* B, float* C,
            int M, int N, int K, const float* bias = nullptr);
    void MatMulTransposedB(cublasHandle_t handle, const float* A, const float* B, float* C,
            int M, int N, int K);
#else
    void MatMul(const float* A, const float* B, float* C,
            int M, int N, int K, const float* bias = nullptr);
    void MatMulTransposedB(const float* A, const float* B, float* C,
            int M, int N, int K);
#endif

    void SoftMax(float *x, int size);
    float DotProd(const float* A, const float* B, int length);
}
