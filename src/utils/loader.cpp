#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>

using namespace std;

//embeddings shape (50257, 768)
constexpr int kVocabSize = 50257;
constexpr int kDimensions = 768;
constexpr size_t kTotalElements = (size_t)kVocabSize*kDimensions;

int main(){
    const char * file_path = "../../data/weights_embeddings.bin";
    vector<float> weights(kTotalElements);
    ifstream file_reader(file_path,ios::binary);
    if(!file_reader){
        fprintf(stderr, "Error: Failed to open file weights_embeddings.bin \n");
        return 1;
    }
    file_reader.read(reinterpret_cast<char*>(weights.data()),sizeof(float)*weights.size());
    if(!file_reader){
        fprintf(stderr, "Error: Failed while reaindg only read %d bytes \n",file_reader.gcount());
        return 1;
    }
    printf("Successfully loaded %d weights \n",kTotalElements);
}
