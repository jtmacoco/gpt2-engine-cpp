#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
#include "ops.hpp"
#include <iostream>
#include <filesystem>
#include <string>
namespace fs = std::filesystem;


int main(int argc, char** argv){
    std::string weights_file = "data/gpt2_embeddings.bin";

    //contains both positional and token weights
    auto weights = WeightsLoader::load_weights(weights_file);

    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    std::string vocab_path  = "data/vocab.json";
    std::string merges_path = "data/merges.txt";

    Tokenizer tokenizer(vocab_path,merges_path);

    std::string input = "The quick brown fox jumps over the lazy";
    int max_tokens_to_generate = 20;
    int generated_count = 0;

    std::vector<int> tokens = tokenizer.Encoder(input);
    std::cout<< "Input: " << input <<std::endl;
    std::vector<int> output_tokens;
    while (generated_count < max_tokens_to_generate){
        int seq_len = tokens.size();
        std::vector<float> input_buffer(seq_len * kModelSize);
        std::vector<float> ln_buffer(seq_len * kModelSize);
        std::vector<float> attention_output(seq_len * kModelSize);
        std::vector<float> ff_output(seq_len * kModelSize);
        std::vector<float> ff_buffer(seq_len * kModelSize * 4);

        InferenceEngine inference_engine(model_weights);
        inference_engine.ApplyEmbedding(tokens,input_buffer.data());

        for (size_t layer_idx = 0; layer_idx < kNumLayers; ++layer_idx){
            std::copy(input_buffer.begin(), input_buffer.end(), ln_buffer.begin());
            for (size_t i = 0; i < seq_len; ++i){
                float* current_token_vec = ln_buffer.data() + (i * kModelSize);
                inference_engine.ApplyLayerNorm(current_token_vec,
                        model_weights.layers[layer_idx].ln_1_beta,
                        model_weights.layers[layer_idx].ln_1_gamma,
                        kModelSize);
            }//end i loop
            inference_engine.AttentionLayer(
                    ln_buffer.data(),
                    attention_output.data(),
                    seq_len,
                    layer_idx);
            for (size_t i = 0; i < seq_len * kModelSize; ++i){
                input_buffer[i] += attention_output[i];
            }//end i loop
            std::copy(input_buffer.begin(), input_buffer.end(), ln_buffer.begin());
            for (size_t i = 0; i < seq_len; ++i){
                float* current_vec = ln_buffer.data() + (i * kModelSize);
                inference_engine.ApplyLayerNorm(current_vec,
                        model_weights.layers[layer_idx].ln_2_beta,
                        model_weights.layers[layer_idx].ln_2_gamma,
                        kModelSize);
            }//end i loop
            inference_engine.FeedForwardLayer(
                    ln_buffer.data(),
                    ff_output.data(),
                    ff_buffer.data(),
                    seq_len,
                    layer_idx);
            for (size_t i = 0; i < seq_len * kModelSize; ++i){
                input_buffer[i] += ff_output[i];
            }
        }//end layer_idx loop
        //std::cout << "FINISHED" << std::endl;

        float* last_token_vec = input_buffer.data() + ((seq_len -1) * kModelSize);
        std::vector<float> final_ln_out(kModelSize);
        std::copy(last_token_vec, last_token_vec + kModelSize, final_ln_out.begin());

        inference_engine.ApplyLayerNorm(final_ln_out.data(),
                model_weights.ln_f_beta,
                model_weights.ln_f_gamma,
                kModelSize);
        std::vector<float> logits(kVocabSize);
        for (size_t v = 0; v < kVocabSize; ++v){
            float* vocab_row = model_weights.weight_token_emb + (v * kModelSize);
            logits[v] = ops::DotProd(final_ln_out.data(), vocab_row, kModelSize);
        }//end v loop

        //Argmax
        int best_token_id = 0;
        float max_logit = logits[0];
        for (size_t v = 1; v < kVocabSize; ++v){
            if (logits[v] > max_logit){
                max_logit = logits[v];
                best_token_id = v;
            }
        }//end v loop
        output_tokens.push_back(best_token_id);
        tokens.push_back(best_token_id);
        std::vector<int> output_vec = {best_token_id};
        std::string pred_output = tokenizer.Decoder(output_vec);
        std::cout << pred_output << std::flush; // std::flush prints it to the screen instantly without waiting for a newline
        if (best_token_id == 50256){
            std::cout<<"End of text token reached"<<std::endl;
            break;
        }
        generated_count++;
    }//end while loop
    std::cout<<"output_tokens size: "<< output_tokens.size()<<std::endl;
    std::string final_output = tokenizer.Decoder(output_tokens);
    std::cout<<"OUTPUT: " << final_output<<std::endl;
    /*
     * uncomment if need to double check
     std::cout << "Predicted Token ID: " << best_token_id << std::endl;
     std::vector<int> output_vec = {best_token_id};
     std::string pred_output = tokenizer.Decoder(output_vec);
     std:: cout<< "Test output: "<< pred_output <<std::endl;
     */

    return 0;
}
