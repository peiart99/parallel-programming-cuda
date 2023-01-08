
#ifndef PARALLEL_PROGRAMMING_CUDA_CALCULATIONS_H
#define PARALLEL_PROGRAMMING_CUDA_CALCULATIONS_H
#include <cstdio>
#include <cuda.h>
#include <iostream>

void calculateAnswer(int data_array_length, int reach, float *data_array, float *out_array);
void calculateSquare(int data_array_length, int reach, int central_i, int central_j, float *data_array, float *out_array);
__global__ void deviceCalculateAnswer(int data_array_length, int reach, int elements_per_thread, int block_size, float *data_array, float *out_array);
__global__ void deviceCalculateAnswer_shared(int data_array_length, int reach, int elements_per_thread, int block_size, int shared_size, int shared_elements_per_thread, float *data_array, float *out_array);

#endif //PARALLEL_PROGRAMMING_CUDA_CALCULATIONS_H
