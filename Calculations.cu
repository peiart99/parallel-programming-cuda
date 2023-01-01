#include "Calculations.h"

void calculateAnswer(int array_length, int reach, float *data_array, float *out_array)
{
    for(int i {0}; i < (array_length - (2 * reach)); i++)
    {
        for(int j {0}; j < (array_length - (2 * reach)); j++)
        {
            calculateSquare(array_length, reach, i, j, data_array, out_array);
        }
    }
}


void calculateSquare(int array_length, int reach, int central_i, int central_j, float *data_array, float *out_array)
{
    for(int i {central_i}; i < 2 * reach + 1 + central_i; i++)
    {
        for(int j {central_j}; j < 2 * reach + 1 + central_j; j++)
        {
            out_array[(central_i * (array_length - (2 * reach))) + central_j] += data_array[(i * array_length) + j];
        }
    }
}

__global__ void deviceCalculateAnswer(int array_length, int reach, int elements_per_thread, float *data_array, float *out_array)
{
    int tid = threadIdx.x * elements_per_thread;
    int bid = threadIdx.y;
    for(int k {0}; k < elements_per_thread; k++)
    {
        for(int i {bid}; i < 2 * reach + 1 + bid; i++)
        {
            for(int j {tid}; j < 2 * reach + 1 + tid; j++)
            {
                out_array[(bid * (array_length - (2 * reach))) + tid] += data_array[(i * array_length) + j];
            }
        }
        tid++;
    }

}