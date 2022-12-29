
#ifndef PARALLEL_PROGRAMMING_CUDA_CALCULATIONS_H
#define PARALLEL_PROGRAMMING_CUDA_CALCULATIONS_H
#include <cstdio>
#include <cuda.h>
#include <iostream>

void calculateAnswer(int array_length, int reach, float *data_array, float *out_array);
void calculateSquare(int array_length, int reach, int central_i, int central_j, float *data_array, float *out_array);
__global__ void deviceCalculateAnswer(int array_length, int reach, float *data_array, float *out_array);

#endif //PARALLEL_PROGRAMMING_CUDA_CALCULATIONS_H
