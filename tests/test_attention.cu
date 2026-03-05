#include <iostream>
#include <vector>
#include <fstream>
#include "inference_engine.hpp"

void save_to_file(const std::vector<float>& vec, const std::string& filename) {
    std::ofstream out(filename);
    for (float val : vec) out << val << "\n";
}

int main() {
    int seq_len = 4;
    int hidden_dim = 768;
    
    std::string weights_file = "data/gpt2_embeddings.bin"; 
    auto weights_vec = WeightsLoader::load_weights(weights_file);
    GPT2Weights model_weights;
    model_weights.map_from_vector(weights_vec);

    InferenceEngine engine(model_weights);

    std::vector<float> h_input(seq_len * hidden_dim, 0.5f); 
    std::vector<float> h_output(seq_len * hidden_dim, 0.0f);

    float *d_input, *d_output;
    cudaMalloc(&d_input, seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_output, seq_len * hidden_dim * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    engine.AttentionLayer(d_input, d_output, seq_len, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output, seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    save_to_file(h_input,"tests/h_input.txt");
    save_to_file(h_output,"tests/h_output.txt");

    
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
