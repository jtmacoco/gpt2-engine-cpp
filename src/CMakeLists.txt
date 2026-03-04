#Common files
set(COMMON_UTILS
    utils/weights_loader.cpp
    utils/tokenizer.cpp
    utils/ops.cpp
)
set(CUDA_UTILS
        utils/weights_loader.cpp
        utils/tokenizer.cpp
        utils/ops.cu
)

#CPU target
add_executable(${PROJECT_NAME}_cpu
        main.cpp
        backend/inference_engine.cpp
        ${COMMON_UTILS}
)

#Cuda target
add_executable(${PROJECT_NAME}_cuda
        main.cu
        backend/inference_engine.cu
        ${CUDA_UTILS}
        
)

target_compile_features(${PROJECT_NAME}_cpu PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME}_cpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE USE_CUDA)
target_link_libraries(${PROJECT_NAME}_cuda PRIVATE CUDA::cublas CUDA::cudart)


target_compile_features(${PROJECT_NAME}_cuda PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME}_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE USE_CUDA)
