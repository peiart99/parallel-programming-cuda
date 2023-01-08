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

__global__ void deviceCalculateAnswer(int array_length, int reach, int elements_per_thread, int block_size, float *data_array, float *out_array)
{

    int k_multiplier {1};
    if(elements_per_thread > 1)
    {
        k_multiplier = elements_per_thread / 2;
    }
    int d_i = threadIdx.y + (blockIdx.y * block_size * k_multiplier);
    int d_j = threadIdx.x + (blockIdx.x * block_size * k_multiplier);
    for(int k_i {0}; k_i < elements_per_thread; k_i++)
    {
            for(int i {d_i}; i < 2 * reach + 1 + d_i; i++)
            {
                for(int j {d_j}; j < 2 * reach + 1 + d_j; j++)
                {
                    out_array[(d_i * (array_length - (2 * reach))) + d_j] += data_array[(i * array_length) + j];
                }
            }

            if(k_i < (k_multiplier - 1))
            {
                d_i += block_size;
            }else if (k_i == (k_multiplier - 1))
            {
                d_i = threadIdx.y + (blockIdx.y * block_size * k_multiplier);
                d_j += block_size;
            }else if(k_i > (k_multiplier - 1))
            {
                d_i += block_size;
            }
    }
}

__global__ void deviceCalculateAnswer_shared(int array_length, int reach, int elements_per_thread, int block_size, int shared_size, int data_move_per_thread, float *data_array, float *out_array)
{

    int k_multiplier {1};
    if(elements_per_thread > 1)
    {
        k_multiplier = elements_per_thread / 2;
    }
    int d_i = threadIdx.y + (blockIdx.y * block_size * k_multiplier);
    int d_j = threadIdx.x + (blockIdx.x * block_size * k_multiplier);

    int k_shmem_modifier_i {0};
    int k_shmem_modifier_j {0};

    int shared_start_i = blockIdx.y * (block_size * k_multiplier);
    int shared_start_j = blockIdx.x * (block_size * k_multiplier);

    int t_id = (block_size * threadIdx.y) + threadIdx.x;

    int elements_counted = t_id * data_move_per_thread;

    int thread_shared_start_i = (elements_counted / shared_size);
    int thread_shared_start_j = (elements_counted % shared_size);

    int counter {0};


    extern __shared__ float shared_data[];


    for(int i {thread_shared_start_i}; i < shared_size; i++)
    {
        for(int j {thread_shared_start_j}; j < shared_size; j++)
        {
            if(counter == data_move_per_thread)
            {
                goto endloop;
            }
            shared_data[(i * shared_size) + j] = data_array[((i + shared_start_i) * array_length) + (j + shared_start_j)];
            counter++;
        }
        thread_shared_start_j = 0;
    }
    endloop:

    __syncthreads();


    for(int k_i {0}; k_i < elements_per_thread; k_i++)
    {
        for(int i = threadIdx.y + k_shmem_modifier_i; i < (2 * reach) + 1 + (threadIdx.y + k_shmem_modifier_i); i++)
        {
            for(int j = (threadIdx.x + k_shmem_modifier_j); j < (2 * reach) + 1 + (threadIdx.x + k_shmem_modifier_j); j++)
            {
                out_array[(d_i * (array_length - (2 * reach))) + d_j] += shared_data[(i * shared_size) + j];
            }
        }

        if(k_i < (k_multiplier - 1))
        {
            d_i += block_size;
            k_shmem_modifier_i += block_size;
        }else if (k_i == (k_multiplier - 1))
        {
            d_i = threadIdx.y + (blockIdx.y * block_size * k_multiplier);
            k_shmem_modifier_i = 0;
            d_j += block_size;
            k_shmem_modifier_j += block_size;
        }else if(k_i > (k_multiplier - 1))
        {
            d_i += block_size;
            k_shmem_modifier_i += block_size;
        }
    }

    __syncthreads();

}