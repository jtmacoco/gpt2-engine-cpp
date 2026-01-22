#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <filesystem>
#include <string>
namespace fs = std::filesystem;

int main(int argc, char** argv){
   std::string weights_file = "data/gpt2_embeddings.bin";
   auto weights = WeightsLoader::load_weights(weights_file);

   std::string vocab_path = "data/vocab.json";
   std::string merges_path = "data/merges.txt";
   Tokenizer tokenizer(vocab_path,merges_path);
   std::vector<int> tokens = tokenizer.Encoder("Hello World");
   for (int i = 0; i < tokens.size(); i++){
       std::cout<< tokens[i] << " "; 
   }
   std::cout<<std::endl;

   std::string text = tokenizer.Decoder(tokens);
   std::cout<< text << std::endl;
   

   return 0;
}
