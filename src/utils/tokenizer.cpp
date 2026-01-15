#include "tokenizer.hpp"
#include "json.hpp"
#include <iostream>
#include <fstream>
Tokenizer::Tokenizer(const std::string& vocab_path, const std::string& merge_path){

    std::ifstream vocab_file(vocab_path, std::ifstream::binary);
    std::ifstream mergers_file(merge_path, std::ifstream::binary);

    if(!vocab_file.is_open()){
        std::cerr<< "Error: Could not open " << vocab_path << std::endl;
        return;
    }
    //read the json file
    nlohmann::json vocab_data;
    vocab_file >> vocab_data;

    //iterate and store vocabulary in unordered_map
    for(auto& [key, value] : vocab_data.items()){
        vocab_[key] = value.get<int>();
        inv_vocab_[value.get<int>()] = key;//needed for decoder
    }

    std::string line;
    //skip first line
    std::getline(mergers_file,line);
    while(std::getline(mergers_file,line)){
        if(line.empty()) continue;

        //find the spaces between two tokens
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos){
            std::string first  = line.substr(0,space_pos);
            std::string second = line.substr(space_pos+1);
            merges_.push_back({first,second});
        }
    }
}

std::vector<int> Tokenizer::Encoder(const std::string& text){
    //TODO
    std::vector<int>dummy;
    return dummy;
}

std:: string Tokenizer::Decoder(const std::vector<int>& tokens){
    //TODO
    std::string dummy = "";
    return dummy;
}





