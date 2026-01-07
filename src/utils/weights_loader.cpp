#include "weights_loader.hpp"
#include <fstream>
#include <cstdio>


namespace WeightsLoader{
    std::vector<float> load_weights(const char *file_path){
        std::vector<float> weights(kTotalElements);
        std::ifstream file_reader(file_path,std::ios::binary);

        if(!file_reader){
            fprintf(stderr, "Error: Failed to open file %s \n",file_path);
            return {};
        }

        file_reader.read(reinterpret_cast<char*>(weights.data()),sizeof(float)*weights.size());

        if(!file_reader){
            fprintf(stderr, "Error: Failed while reaindg only read %ld bytes \n",file_reader.gcount());
            return {};
        }
        //printf("Successfully loaded %zu weights \n",kTotalElements);
        return weights;
    }
}