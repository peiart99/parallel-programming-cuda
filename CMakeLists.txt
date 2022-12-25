cmake_minimum_required(VERSION 3.23)
project(parallel_programming_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(parallel_programming_cuda main.cu Array.cu Array.h Generator.cu Generator.h Calculations.cu Calculations.cuh)

set_target_properties(parallel_programming_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
