#include "ops.hpp"
#include <cstddef>
#include <cmath>
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
}
