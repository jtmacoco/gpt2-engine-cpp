add_executable(test_attention test_attention.cu)

target_link_libraries(test_attention PRIVATE gpt2_cuda_core)

add_test(NAME AttentionTest COMMAND test_attention)
