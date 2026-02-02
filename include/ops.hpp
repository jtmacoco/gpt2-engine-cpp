namespace ops{
    void MatMul(const float* A, const float* B, float* C,
                int M, int N, int K, const float* bias = nullptr);
}
