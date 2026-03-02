#Common files
set(COMMON_UTILS
    utils/weights_loader.cpp
    utils/tokenizer.cpp
    utils/ops.cpp
)
#CPU target
add_executable(${PROJECT_NAME}_cpu
        main.cpp
        backend/inference_engine.cpp
        ${COMMON_UTILS}
)

target_compile_features(${PROJECT_NAME}_cpu PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME}_cpu PRIVATE ${PROJECT_SOURCE_DIR}/include)

#Cuda target
add_executable(${PROJECT_NAME}_cuda
        sandbox.cu
)

target_compile_features(${PROJECT_NAME}_cuda PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME}_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE USE_CUDA)
