#ifndef PARALLEL_PROGRAMMING_CUDA_ARRAY_H
#define PARALLEL_PROGRAMMING_CUDA_ARRAY_H
#include <iostream>
#include "Generator.h"

void printArray(int array_length, float *array);
void populateArray(int array_length, float *array, bool debug);
int compareArrays(int array_length, float *a, float *b);

#endif //PARALLEL_PROGRAMMING_CUDA_ARRAY_H
