#include "weights_loader.hpp"
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

int main(int argc, char** argv){
   fs::path weights_file;
   if (argc > 1)
      weights_file = argv[1];
   else{
      fprintf(stderr,"Missing argument .bin path \n");
      return 1;
   }
   if(!fs::exists(weights_file)){
      fprintf(stderr,"Weights file not found: %s \n",weights_file.c_str());
      return 1;
   }

   auto weights = WeightsLoader::load_weights(weights_file.c_str());
   return 0;
}
