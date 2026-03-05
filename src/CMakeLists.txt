add_library(gpt2_cpu_core STATIC
    utils/weights_loader.cpp
    utils/tokenizer.cpp
    utils/ops.cpp
    backend/inference_engine.cpp
)
target_include_directories(gpt2_cpu_core PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(gpt2_cuda_core STATIC
    utils/weights_loader.cpp
    utils/tokenizer.cpp
    utils/ops.cu
    backend/inference_engine.cu
)
target_include_directories(gpt2_cuda_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(gpt2_cuda_core PUBLIC USE_CUDA)
target_link_libraries(gpt2_cuda_core PUBLIC CUDA::cublas CUDA::cudart)

add_executable(${PROJECT_NAME}_cpu main.cpp)
target_link_libraries(${PROJECT_NAME}_cpu PRIVATE gpt2_cpu_core)

add_executable(${PROJECT_NAME}_cuda main.cu)
target_link_libraries(${PROJECT_NAME}_cuda PRIVATE gpt2_cuda_core)
