#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP
#include <unordered_map>
#include <iostream>
#include <vector>
#include <string>

class Tokenizer{
    public:
        Tokenizer(const std::string& vocab_path, const std::string& merge_path);

        std::vector<int> Encoder(const std::string& text);
        std::string Decoder(const std::vector<int>& tokens);

    private:
        std::unordered_map<std::string, int> vocab_;
        std::unordered_map<int, std::string> inv_vocab_;
        std::vector<std::pair<std::string, std::string>> merges_;
};
#endif
