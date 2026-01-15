
add_executable(${PROJECT_NAME}_run)

target_compile_features(${PROJECT_NAME}_run PRIVATE cxx_std_20)

target_sources(${PROJECT_NAME}_run
    PRIVATE
    main.cpp
    utils/weights_loader.cpp
    utils/tokenizer.cpp
)

target_compile_features(${PROJECT_NAME}_run PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME}_run PRIVATE ../include)
