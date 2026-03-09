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
        const float beta  = 0.0f;

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
            std::cerr << "cuBLass matrix mult failed" << std::endl;

        if (d_bias != nullptr){
            int total_elements = M * N;
            int threads_per_block = 256;
            int blocks_per_grid = (total_elements + threads_per_block -1) / threads_per_block;
            cudaStream_t stream;
            cublasGetStream(handle, &stream);
            AddBiasKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_C, d_bias, M, N);
        }
    }
    void MatMulTransposedB(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int M, int N, int K){
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasStatus_t status = cublasSgemm(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                d_B,
                K,
                d_A,
                K,
                &beta,
                d_C,
                N
                );
        if (status != CUBLAS_STATUS_SUCCESS){
            std::cerr << "cuBLASS Transpose error" << std::endl;
        }
    }
    __global__ void UnfusedQKVKernel(const float* qkv_buffer, float* Q, float* K, float* V, 
            int seq_len, int num_heads, int head_dim){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int hidden_dim = num_heads * head_dim;
        int total_elements = seq_len * hidden_dim;
        if (idx < total_elements){
            int d = idx % head_dim;
            int h = (idx / head_dim) % num_heads;
            int s = idx / hidden_dim;

            int src_idx = s * (3 * hidden_dim) + (h * head_dim + d);
            int dst_idx = h * (seq_len * head_dim) + s * head_dim + d;

            Q[dst_idx] = qkv_buffer[src_idx];
            K[dst_idx] = qkv_buffer[src_idx + hidden_dim];
            V[dst_idx] = qkv_buffer[src_idx + (hidden_dim * 2)];
        }

    }
    __global__ void ScaledAndMaskKernel(float* score_head, int seq_len, float scale){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = seq_len * seq_len;
        if (idx < total_elements){
            int row = idx / seq_len;
            int col = idx % seq_len;
            if (col > row){
                score_head[idx] = -1e9f;
            }
            else{
                score_head[idx] *=scale;
            }
        }
    }
    __global__ void AttentionSoftMaxKernel(float* scores, int row_length) {
        int row = blockIdx.x; 
        int tid = threadIdx.x;

        //Jump to the correct row using row_length instead of seq_len
        float* row_ptr = scores + (row * row_length);

        __shared__ float s_max;
        __shared__ float s_sum;

        float thread_max = -1e9f;
        for (int i = tid; i < row_length; i += blockDim.x) { 
            if (row_ptr[i] > thread_max) {
                thread_max = row_ptr[i];
            }
        }

        __shared__ float max_array[256]; 
        max_array[tid] = thread_max;
        __syncthreads(); 

        if (tid == 0) {
            float row_max = -1e9f;
            for (int i = 0; i < blockDim.x; ++i) {
                if (max_array[i] > row_max) row_max = max_array[i];
            }
            s_max = row_max;
        }
        __syncthreads();

        float thread_sum = 0.0f;
        for (int i = tid; i < row_length; i += blockDim.x) { // changed seq_len to row_length
            row_ptr[i] = expf(row_ptr[i] - s_max);
            thread_sum += row_ptr[i];
        }

        __shared__ float sum_array[256];
        sum_array[tid] = thread_sum;
        __syncthreads();

        if (tid == 0) {
            float total_sum = 0.0f;
            for (int i = 0; i < blockDim.x; ++i) {
                total_sum += sum_array[i];
            }
            s_sum = total_sum;
        }
        __syncthreads();

        //Normalize 
        for (int i = tid; i < row_length; i += blockDim.x) { // changed seq_len to row_length
            row_ptr[i] /= s_sum;
        }
    }

    void AttentionSoftMax(float* d_scores, int num_rows, int row_length, cudaStream_t stream) {
        int blocks = num_rows; // 1 block per row
        int threads = 256;

        AttentionSoftMaxKernel<<<blocks, threads, 0, stream>>>(d_scores, row_length);
    }
    __global__ void ConcatHeadsKernel(const float* context_layer, float* proj_output,
            int seq_len, int num_heads, int head_dim ){

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int hidden_dim = num_heads * head_dim;
        int total_elements = seq_len * hidden_dim;

        if(idx < total_elements){
            int d = idx % head_dim;
            int h = (idx / head_dim) % num_heads;
            int s = idx /  hidden_dim;

            int src_idx = h * (seq_len * head_dim ) + s * head_dim + d;
            proj_output[idx] = context_layer[src_idx];
        }
    }

    __global__ void GeluKernel(float* buffer, int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size){
            float x = buffer[idx];
            float x_cubed = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x_cubed);

            buffer[idx] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }

    void Gelu(float* d_buffer, int size, cudaStream_t stream){
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        GeluKernel<<<blocks, threads, 0, stream>>>(d_buffer, size);
    }

    __global__ void LayerNormKernel(float* x, const float* beta, const float* gamma, int dim){
        int tid = threadIdx.x;

        __shared__ float s_sum[256];
        __shared__ float s_sum_sq[256];

        __shared__ float s_mean;
        __shared__ float s_std_dev;

        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;

        for (size_t i = tid; i < dim; i += blockDim.x){
            float val =  x[i];
            local_sum += val;
            local_sum_sq += val * val;
        }

        s_sum[tid] = local_sum;
        s_sum_sq[tid] = local_sum_sq;
        __syncthreads();

        if (tid == 0){
            float total_sum = 0.0f;
            float total_sum_sq = 0.0f;

            for (size_t i = 0; i < blockDim.x; ++i){
                total_sum += s_sum[i];
                total_sum_sq += s_sum_sq[i];
            }//end i loop

            float mean = total_sum / dim;
            float var = (total_sum_sq / dim) - (mean * mean);

            s_mean = mean;
            s_std_dev = sqrtf(var + 1e-5f);
        }// end if
        __syncthreads();
        for (size_t i = tid; i < dim; i += blockDim.x){
            float norm = (x[i] - s_mean) / s_std_dev;
            x[i] = (norm * gamma[i]) + beta[i];
        }
    }

    void LayerNorm(float* d_x, const float* d_beta, const float* d_gamma, int dim, cudaStream_t stream){
        int threads = 256;
        int blocks = 1;
        LayerNormKernel<<<blocks, threads, 0, stream>>>(d_x, d_beta, d_gamma, dim);
    }
    __global__ void AddResidualKernel(float* target, const float* source, int size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size){
            target[idx] += source[idx];
        }
    }
    void AddResidual(float* d_target, const float* d_source, int size, cudaStream_t stream){
        int threads = 256;
        int blocks = (size + threads-1) / threads;
        AddResidualKernel<<<blocks, threads, 0, stream>>>(d_target, d_source, size);
    }
    __global__ void EmbeddingKernel(const int* tokens, const float* token_emb, const float* pos_emb, float* output,
            int seq_len, int hidden_dim, int current_pos) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = seq_len * hidden_dim;

        if (idx < total_elements) {
            int t = idx / hidden_dim; // Local index (0 to seq_len - 1)
            int d = idx % hidden_dim;

            int token_id = tokens[t];
            int absolute_pos = current_pos + t; // Add current_pos to get the true position!

            output[idx] = token_emb[token_id * hidden_dim + d] + pos_emb[absolute_pos * hidden_dim + d];
        }
    }

    void ApplyEmbedding(const int* d_tokens, const float* d_token_emb, const float* d_pos_emb, float* d_output,
            int seq_len, int hidden_dim, int current_pos, cudaStream_t stream) {
        int total_elements = seq_len * hidden_dim;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        EmbeddingKernel<<<blocks, threads, 0, stream>>>(d_tokens, d_token_emb, d_pos_emb, d_output, seq_len, hidden_dim, current_pos);
    }
    __global__ void AppendKVKernel(
            const float* new_K, const float* new_V,
            float* K_cache, float* V_cache,
            int current_pos, int max_seq_len, int num_heads, int head_dim){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int total_elements = num_heads * head_dim;

        if (idx < total_elements){
            int d = idx % head_dim;
            int h = idx / head_dim;

            int src_idx = h * head_dim + d;
            int dst_idx = h * (max_seq_len * head_dim) + (current_pos * head_dim) + d;
            K_cache[dst_idx] = new_K[src_idx];
            V_cache[dst_idx] = new_V[src_idx];
        }

    }
    void AppendKV(
            const float* d_new_K, const float* d_new_V,
            float* d_K_cache, float* d_V_cache,
            int current_pos, int max_seq_len, int num_heads, int head_dim, cudaStream_t stream){
        int total_elements = num_heads * head_dim;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        AppendKVKernel<<<blocks, threads, 0, stream>>>(
                d_new_K, d_new_V,
                d_K_cache, d_V_cache,
                current_pos, max_seq_len, num_heads, head_dim
                );
    }
    __global__ void ScaledKernel(float* scores, int total_elements, float scale){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements){
            scores[idx] *= scale;
        }
    }


    __global__ void BulkAppendKVKernel(
            const float* new_K, const float* new_V,
            float* K_cache, float* V_cache,
            int seq_len, int max_seq_len, int num_heads, int head_dim
            ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = seq_len * num_heads * head_dim;

        if (idx < total_elements) {
            int d = idx % head_dim;
            int s = (idx / head_dim) % seq_len;
            int h = idx / (seq_len * head_dim);

            int src_idx = idx; 
            int dst_idx = h * (max_seq_len * head_dim) + s * head_dim + d;

            K_cache[dst_idx] = new_K[src_idx];
            V_cache[dst_idx] = new_V[src_idx];
        }
    }

    void BulkAppendKV(
            const float* d_new_K, const float* d_new_V,
            float* d_K_cache, float* d_V_cache,
            int seq_len, int max_seq_len, int num_heads, int head_dim,
            cudaStream_t stream
            ) {
        int total_elements = seq_len * num_heads * head_dim;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        BulkAppendKVKernel<<<blocks, threads, 0, stream>>>(
                d_new_K, d_new_V,
                d_K_cache, d_V_cache,
                seq_len, max_seq_len, num_heads, head_dim
                );

    }
    __global__ void ArgMaxKernel(const float* logits, int n, int* out_idx) {
        __shared__ float s_val[256];
        __shared__ int   s_idx[256];

        int tid = threadIdx.x;
        float best_v = -1e30f;
        int best_i = 0;

        for (int i = tid; i < n; i += blockDim.x) {
            float v = logits[i];
            if (v > best_v) { best_v = v; best_i = i; }
        }

        s_val[tid] = best_v;
        s_idx[tid] = best_i;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                float v2 = s_val[tid + offset];
                int   i2 = s_idx[tid + offset];
                if (v2 > s_val[tid]) {
                    s_val[tid] = v2;
                    s_idx[tid] = i2;
                }
            }
            __syncthreads();
        }

        if (tid == 0) *out_idx = s_idx[0];
    }

    void ArgMax(const float* d_logits, int n, int* d_out_idx, cudaStream_t stream) {
        ArgMaxKernel<<<1, 256, 0, stream>>>(d_logits, n, d_out_idx);
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
