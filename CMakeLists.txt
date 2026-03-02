cmake_minimum_required(VERSION 4.2.1)

project(gpt2_inference_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

add_subdirectory(src)

