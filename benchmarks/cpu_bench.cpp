#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include "inference_engine.hpp"
#include "ops.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

struct RunStats {
    double total_ms = 0.0;
    double ttft_ms  = 0.0;
    int generated   = 0;   // generated tokens (includes first)
    int prompt_len  = 0;
};

static RunStats run_once_cpu(
    InferenceEngine& inference_engine,
    const GPT2Weights& model_weights,
    const std::vector<int>& prompt_tokens,
    int tokens_to_generate
) {
    using clock = std::chrono::steady_clock;

    std::vector<int> tokens = prompt_tokens;
    const int prompt_len = (int)tokens.size();

    // Allocate buffers once per run (NOT inside the token loop)
    // Note: seq_len grows each token in this CPU version (no KV cache), so allocate max needed.
    const int max_seq_len = prompt_len + tokens_to_generate;
    std::vector<float> input_buffer((size_t)max_seq_len * (size_t)kModelSize);
    std::vector<float> ln_buffer((size_t)max_seq_len * (size_t)kModelSize);
    std::vector<float> attention_output((size_t)max_seq_len * (size_t)kModelSize);
    std::vector<float> ff_output((size_t)max_seq_len * (size_t)kModelSize);
    std::vector<float> ff_buffer((size_t)max_seq_len * (size_t)kModelSize * 4ull);

    std::vector<float> final_ln_out((size_t)kModelSize);
    std::vector<float> logits((size_t)kVocabSize);

    auto start_total = clock::now();
    auto stop_prefill = start_total;

    int generated_count = 0;
    bool prefill_recorded = false;

    while (generated_count < tokens_to_generate) {
        const int seq_len = (int)tokens.size();

        float* input_ptr = input_buffer.data();
        float* ln_ptr = ln_buffer.data();
        float* attn_ptr = attention_output.data();
        float* ff_out_ptr = ff_output.data();
        float* ff_buf_ptr = ff_buffer.data();

        // Forward pass
        inference_engine.ApplyEmbedding(tokens, input_ptr);

        for (size_t layer_idx = 0; layer_idx < kNumLayers; ++layer_idx) {
            // LN #1 (copy -> ln buffer)
            std::copy(input_ptr, input_ptr + (size_t)seq_len * (size_t)kModelSize, ln_ptr);

            for (int i = 0; i < seq_len; ++i) {
                float* vec = ln_ptr + (size_t)i * (size_t)kModelSize;
                inference_engine.ApplyLayerNorm(
                    vec,
                    model_weights.layers[layer_idx].ln_1_beta,
                    model_weights.layers[layer_idx].ln_1_gamma,
                    kModelSize
                );
            }

            inference_engine.AttentionLayer(
                ln_ptr,
                attn_ptr,
                (size_t)seq_len,
                layer_idx
            );

            // Residual add
            for (size_t i = 0; i < (size_t)seq_len * (size_t)kModelSize; ++i)
                input_ptr[i] += attn_ptr[i];

            // LN #2
            std::copy(input_ptr, input_ptr + (size_t)seq_len * (size_t)kModelSize, ln_ptr);

            for (int i = 0; i < seq_len; ++i) {
                float* vec = ln_ptr + (size_t)i * (size_t)kModelSize;
                inference_engine.ApplyLayerNorm(
                    vec,
                    model_weights.layers[layer_idx].ln_2_beta,
                    model_weights.layers[layer_idx].ln_2_gamma,
                    kModelSize
                );
            }

            inference_engine.FeedForwardLayer(
                ln_ptr,
                ff_out_ptr,
                ff_buf_ptr,
                (size_t)seq_len,
                layer_idx
            );

            // Residual add
            for (size_t i = 0; i < (size_t)seq_len * (size_t)kModelSize; ++i)
                input_ptr[i] += ff_out_ptr[i];
        }

        // Final layer norm on last token
        float* last_token_vec = input_ptr + (size_t)(seq_len - 1) * (size_t)kModelSize;
        std::copy(last_token_vec, last_token_vec + kModelSize, final_ln_out.begin());

        inference_engine.ApplyLayerNorm(
            final_ln_out.data(),
            model_weights.ln_f_beta,
            model_weights.ln_f_gamma,
            kModelSize
        );

        // CPU logits: dot(final_ln_out, token_embedding_row)
        // (This is extremely expensive; that's fine if you're measuring full CPU baseline.)
        for (size_t v = 0; v < (size_t)kVocabSize; ++v) {
            float* vocab_row = model_weights.weight_token_emb + (v * (size_t)kModelSize);
            logits[v] = ops::DotProd(final_ln_out.data(), vocab_row, kModelSize);
        }

        // Argmax
        int best_token_id = 0;
        float max_logit = logits[0];
        for (size_t v = 1; v < (size_t)kVocabSize; ++v) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                best_token_id = (int)v;
            }
        }

        tokens.push_back(best_token_id);
        generated_count++;

        // Record TTFT after first generated token is produced
        if (!prefill_recorded) {
            stop_prefill = clock::now();
            prefill_recorded = true;
        }

        if (best_token_id == 50256) break; // EOS
    }

    auto stop_total = clock::now();

    std::chrono::duration<double, std::milli> total_dur_ms = stop_total - start_total;
    std::chrono::duration<double, std::milli> ttft_dur_ms  = stop_prefill - start_total;

    RunStats s;
    s.total_ms = total_dur_ms.count();
    s.ttft_ms  = ttft_dur_ms.count();
    s.generated = generated_count;
    s.prompt_len = prompt_len;
    return s;
}

int main(int argc, char** argv) {
    std::string weights_file = "data/gpt2_embeddings.bin";
    auto weights = WeightsLoader::load_weights(weights_file);
    GPT2Weights model_weights;
    model_weights.map_from_vector(weights);

    std::string vocab_path  = "data/vocab.json";
    std::string merges_path = "data/merges.txt";
    Tokenizer tokenizer(vocab_path, merges_path);

    std::string input = "The quick brown fox jumps over the lazy dog and runs into the forest";
    int tokens_to_generate = 128;
    if (argc > 1) tokens_to_generate = std::stoi(argv[1]);

    std::vector<int> prompt_tokens = tokenizer.Encoder(input);

    InferenceEngine inference_engine(model_weights);

    // Warmup + measure
    const int WARMUP_RUNS  = 2;
    const int MEASURE_RUNS = 10;

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        (void)run_once_cpu(inference_engine, model_weights, prompt_tokens, tokens_to_generate);
    }

    std::vector<RunStats> runs;
    runs.reserve(MEASURE_RUNS);
    for (int i = 0; i < MEASURE_RUNS; ++i) {
        runs.push_back(run_once_cpu(inference_engine, model_weights, prompt_tokens, tokens_to_generate));
    }

    // Aggregate mean
    double mean_total_ms = 0.0;
    double mean_ttft_ms  = 0.0;
    double mean_tpot_ms  = 0.0;
    double mean_thr      = 0.0;
    double mean_gen      = 0.0;

    for (const auto& r : runs) {
        mean_total_ms += r.total_ms;
        mean_ttft_ms  += r.ttft_ms;

        const int decode_tokens = r.generated - 1;
        const double decode_ms = r.total_ms - r.ttft_ms;
        const double tpot_ms = (decode_tokens > 0) ? (decode_ms / (double)decode_tokens) : 0.0;
        mean_tpot_ms += tpot_ms;

        const double total_s = r.total_ms / 1000.0;
        const double thr = (total_s > 0.0) ? ((double)r.generated / total_s) : 0.0;
        mean_thr += thr;

        mean_gen += (double)r.generated;
    }

    mean_total_ms /= (double)runs.size();
    mean_ttft_ms  /= (double)runs.size();
    mean_tpot_ms  /= (double)runs.size();
    mean_thr      /= (double)runs.size();
    mean_gen      /= (double)runs.size();

    const int prompt_len = (int)prompt_tokens.size();

    std::cout << "\n==================================================\n";
    std::cout << " CPU BENCHMARK (mean over " << MEASURE_RUNS
              << " runs, warmup " << WARMUP_RUNS << ")\n";
    std::cout << " prompt_len=" << prompt_len
              << ", requested_gen=" << tokens_to_generate << "\n";
    std::cout << "==================================================\n";
    std::cout << "| Prompt | Req Gen | Avg Gen |   TTFT (ms) |   TPOT (ms/tok) |   Total (ms) |   Throughput (tok/s) |\n";
    std::cout << "|-------:|--------:|--------:|------------:|----------------:|-------------:|---------------------:|\n";
    std::cout << "| " << std::setw(6) << prompt_len
              << " | " << std::setw(7) << tokens_to_generate
              << " | " << std::setw(7) << std::fixed << std::setprecision(1) << mean_gen
              << " | " << std::setw(11) << std::fixed << std::setprecision(2) << mean_ttft_ms
              << " | " << std::setw(14) << std::fixed << std::setprecision(2) << mean_tpot_ms
              << " | " << std::setw(11) << std::fixed << std::setprecision(2) << mean_total_ms
              << " | " << std::setw(20) << std::fixed << std::setprecision(2) << mean_thr
              << " |\n";

    return 0;
}
