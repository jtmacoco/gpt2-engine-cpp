#include "tokenizer.hpp"
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>

std::unordered_map<unsigned char, std::string> Tokenizer::GetBytesToUnicode(){
    std::unordered_map<unsigned char, std::string> byte_encoder;

    std::vector<int> bs;
    for (int b = 0x21; b <= 0x7E; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;

    for (int b = 0; b < 256; ++b){
        if (std::find(bs.begin(), bs.end(), b) == bs.end()){
            bs.push_back(b);
            cs.push_back(256+n);
            n++;
        }
    }

    for(size_t i = 0; i < bs.size(); ++i){
        char32_t codepoint = cs[i];
        std::string utf8_char;
        if (codepoint < 0x80){
            utf8_char += (char) codepoint;
        } 
        else if (codepoint < 0x800){
            utf8_char += (char)(0xC0 | (codepoint >> 6));
            utf8_char += (char)(0x80 | (codepoint & 0x3F));
        }
        byte_encoder[(unsigned char) bs[i]] = utf8_char;
    }

    return byte_encoder;

}

std::unordered_map<std::string, unsigned char> Tokenizer::GetUnicodeToBytes(){

    std::unordered_map<std::string, unsigned char> unicode_decoder;

    for (const auto& pair : byte_encoder_){
        unicode_decoder[pair.second] = pair.first;
    }
    return unicode_decoder;
}

Tokenizer::Tokenizer(const std::string& vocab_path, const std::string& merge_path){

    std::ifstream vocab_file(vocab_path, std::ifstream::binary);
    std::ifstream mergers_file(merge_path, std::ifstream::binary);

    if (!vocab_file.is_open()){
        std::cerr<< "Error: Could not open " << vocab_path << std::endl;
        return;
    }
    //read the vocab.json file
    nlohmann::json vocab_data;
    vocab_file >> vocab_data;

    //iterate and store vocabulary in unordered_map
    for (auto& [key, value] : vocab_data.items()){
        vocab_map_[key] = value.get<int>();
        inv_vocab_map_[value.get<int>()] = key;//needed for decoder
    }//end for

    std::string line;

    //rank used to maintain order
    int rank = 0;

    //skip first line since it's "version #"
    std::getline(mergers_file,line);

    while (std::getline(mergers_file,line)){
        if(line.empty()) continue;

        //find the spaces between two tokens
        size_t space_pos = line.find(' ');

        merges_map_[line] = rank;
        rank++;
    }//end while

    byte_encoder_ = GetBytesToUnicode();
    byte_decoder_ = GetUnicodeToBytes();

}//end Tokenizer constructor

//Byte Pair Encoding
std::vector<int> Tokenizer::Encoder(const std::string& text){
    std::list<std::string> encoded_text;

    //convert input text to bye-encoding for weird ascii
    for (unsigned char ch : text){
        encoded_text.push_back(byte_encoder_[ch]);
    }

    //Iteratively find the highes priority pair (lowest rank index) and merge
    while (true){
        std::list<std::string>::iterator best_pair_it = encoded_text.end();
        int best_rank = std::numeric_limits<int>::max();

        auto it = encoded_text.begin();
        auto next_it = std::next(it);

        //Scan tokens to find best adjacent pair to merge
        while (next_it != encoded_text.end()){
            std::string pair_to_check = *it + " " + *next_it;
            auto map_it = merges_map_.find(pair_to_check);

            if (map_it != merges_map_.end()){
                int rank = map_it -> second;
                if (rank < best_rank){
                    best_rank = rank;
                    best_pair_it = it;
                }
            }
            it++;
            next_it++;
        }
        if (best_pair_it == encoded_text.end()) break;

        //merge pair
        auto next_token = std::next(best_pair_it);
        *best_pair_it += *next_token;

        encoded_text.erase(next_token);

    }
    std::vector<int> tokens;
    for (const auto& word : encoded_text){
        if(vocab_map_.count(word)){
            tokens.push_back(vocab_map_[word]);
        }
        else {
            std::cerr << "Token not found in vocab: " << word << std::endl;
        }
    }
    return tokens;
}//end Tokenizer

std:: string Tokenizer::Decoder(const std::vector<int>& tokens){
    std::string bpe_text = "";

    //concat all tokens into one raw string
    for (size_t i = 0; i < tokens.size(); ++i){
        bpe_text += inv_vocab_map_[tokens[i]];
    }

    std::string decoded_text = "";
    //iterate over the BPE string to convert to original bytes
    for (size_t i = 0; i < bpe_text.size();){

        //try to match 1-byte character (OG ASCII)
        std::string one_byte = bpe_text.substr(i,1);

        //try to match 2-byte character (Special chars like Ġ)
        //check bounds to ensure no segfault at end of string
        std::string two_bytes = (i + 1 < bpe_text.length()) ? bpe_text.substr(i,2) : "";

        if (!two_bytes.empty() && byte_decoder_.count(two_bytes)){
            //found special character so add a space
            decoded_text += byte_decoder_.at(two_bytes);
            i += 2;
        }
        else if (byte_decoder_.count(one_byte)){
            decoded_text += byte_decoder_.at(one_byte);
            i+=1;
        }
        else{//fallback 
            decoded_text += bpe_text[i];
            i+=1;
        }
    }
    return decoded_text;
}





