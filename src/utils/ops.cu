#include "ops.hpp"
#include <iostream>
#include <cstddef>
#include <cmath>
#include <cuda.h>
namespace ops{
    __global__ void AddBiasKernel(float* C, const float* bias, int M, int N){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = M * N;
        if (idx < total_elements){
            int col = idx % N;
            if (bias != nullptr){
                C[idx] += bias[col];
            }
        }
    }
    void MatMul(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int M, int N, int K, const float* d_bias){
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasStatus_t status = cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                d_B,
                N,
                d_A,
                K,
                &beta,
                d_C,
                N
        );

        if (status != CUBLAS_STATUS_SUCCESS)
            std::cerr << "cuBLass failed bro" << std::endl;

        if (d_bias != nullptr){
            int total_elements = M * N;
            int threads_per_block = 256;
            int blocks_per_grid = (total_elements + threads_per_block -1) / threads_per_block;
            AddBiasKernel<<<blocks_per_grid, threads_per_block>>>(d_C, d_bias, M, N);
            cudaDeviceSynchronize();
        }
    }
    void MatMulTransposedB(cublasHandle_t handle, const float* A, const float* B, float* C, int M, int N, int K){
        for (size_t i = 0; i < M; ++i){
            //accumulate multiplication
            for (size_t j = 0; j < N; ++j){
                C[i * N + j] = DotProd(&A[i * K], &B[j * K], K);
            }//end k loop
        }//end i loop
    }

    void SoftMax(float* x, int size){
        float max_val = x[0];
        //find max value
        for (size_t i = 1; i < size; ++i){
            if (x[i] > max_val)
                max_val = x[i];
        }
        float sum = 0.0f;
        //calculate the exp sum
        for (size_t i = 0; i < size; ++i){
            x[i] = expf(x[i]-max_val);
            sum+=x[i];
        }
        //Normalize Probabilities
        for(size_t i = 0; i < size; ++i) x[i]/=sum;
    }
    float DotProd(const float* A, const float* B, int length){
        float sum = 0.0f;
        for (size_t i = 0; i < length; ++i){
            sum+= A[i] * B[i];
        }
        return sum;
    }
}
