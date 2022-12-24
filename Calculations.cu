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
    int upper_left_index_j {0};
    int upper_left_index_i {0};
    int square_length = 2 * reach + 1;
    int lower_edge {0};
    int right_edge {0};
    int diff_j {0};
    int diff_i {0};
    if(central_j - reach >= 0)
    {
        upper_left_index_j = central_j - reach;
    }else
    {
        upper_left_index_j = 0;
    }
    diff_j = upper_left_index_j - (central_j - reach);
    right_edge = upper_left_index_j - diff_j + square_length;
    if(right_edge > array_length)
    {
        right_edge = array_length;
    }

    if(central_i - reach >= 0)
    {
        upper_left_index_i = central_i - reach;
    }else
    {
        upper_left_index_i - 0;
    }
    diff_i = upper_left_index_i - (central_i - reach);
    lower_edge = upper_left_index_i - diff_i + square_length;

    for(int i {upper_left_index_i}; i < lower_edge; i++)
    {
        for(int j {upper_left_index_j}; j < right_edge; j++)
        {
            out_array[(central_i * (array_length - (2 * reach))) + central_j] += data_array[(i * array_length) + j];
        }
    }
}

__global__ void deviceCalculateAnswer(int array_length, int reach, float *data_array, float *out_array)
{
    //printf("block id: %d thread id: %d\n", blockIdx.x, threadIdx.x);
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int upper_left_index_j {0};
    int upper_left_index_i {0};
    int square_length = 2 * reach + 1;
    int lower_edge {0};
    int right_edge {0};
    int diff_j {0};
    int diff_i {0};
    if(tid - reach >= 0)
    {
        upper_left_index_j = tid - reach;
    }else
    {
        upper_left_index_j = 0;
    }
    diff_j = upper_left_index_j - (tid - reach);
    right_edge = upper_left_index_j - diff_j + square_length;
    if(right_edge > array_length)
    {
        right_edge = array_length;
    }

    if(bid - reach >= 0)
    {
        upper_left_index_i = bid - reach;
    }else
    {
        upper_left_index_i - 0;
    }
    diff_i = upper_left_index_i - (bid - reach);
    lower_edge = upper_left_index_i - diff_i + square_length;

    for(int i {upper_left_index_i}; i < lower_edge; i++)
    {
        for(int j {upper_left_index_j}; j < right_edge; j++)
        {
            out_array[(bid * (array_length - (2 * reach))) + tid] += data_array[(i * array_length) + j];
        }
    }
}