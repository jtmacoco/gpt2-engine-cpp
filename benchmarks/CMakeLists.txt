add_executable(cpu_bench cpu_bench.cpp)
add_executable(gpu_bench gpu_bench.cu)


target_link_libraries(cpu_bench PRIVATE gpt2_cpu_core)
target_link_libraries(gpu_bench PRIVATE gpt2_cuda_core)


add_test(NAME GPUBENCH COMMAND gpu_bench)
add_test(NAME CUPBENCH COMMAND cpu_bench)
