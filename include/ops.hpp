namespace ops{
    void MatMul(const float* A, const float* B, float* C,
                int M, int N, int K, const float* bias = nullptr);
    void MatMulTransposedB(const float* A, const float* B, float* C,
                int M, int N, int K);

    void SoftMax(float *x, int size);
    float DotProd(const float* A, const float* B, int length);
}
