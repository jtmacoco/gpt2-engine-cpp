#include "weights_loader.hpp"
#include <fstream>
#include <cstdio>
#include <iostream>


namespace WeightsLoader{
    std::vector<float> load_weights(const std::string file_path){
        std::vector<float> weights(kTotalElements);
        std::ifstream file_reader(file_path,std::ios::binary);

        if(!file_reader){
            std::cerr<< "Error: Failed to open file " << file_path << std::endl;
            return {};
        }

        file_reader.read(reinterpret_cast<char*>(weights.data()),sizeof(float)*weights.size());

        if(!file_reader){
            std::cerr << "Error: Failed while reading only read bytes " << file_reader.gcount() << "bytes" << std::endl;
            return {};
        }
        //printf("Successfully loaded %zu weights \n",kTotalElements);
        return weights;
    }
}
