#include "ops.hpp"
#include <cstddef>
namespace ops{
    void MatMul(const float* A, const float* B, float* C, int M, int N, int K, const float* bias){
        for (size_t i = 0; i < M; ++i){
            //init row with bias 
            for (size_t j = 0; j < N; ++j){
                C[i * N + j] = (bias != nullptr) ? bias[j] : 0.0f;
            }
            //accumulate multiplication
            for (size_t k = 0; k < K; ++k){
                float a_val = A[i * K + k];//row value
                for (size_t j = 0; j < N; ++j){
                    C[i * N + j] += a_val * B[k * N + j];
                }//end j loop
            }//end k loop
        }//end i loop
    }
}
