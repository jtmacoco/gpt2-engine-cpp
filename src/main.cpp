#include "weights_loader.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <filesystem>
#include <string>
namespace fs = std::filesystem;

int main(int argc, char** argv){
   std::string weights_file = "data/weights_embeddings.bin";
   auto weights = WeightsLoader::load_weights(weights_file);

   std::string vocab_path = "data/vocab.json";
   std::string merges_path = "data/merges.txt";
   Tokenizer tokenizer(vocab_path,merges_path);

   return 0;
}
