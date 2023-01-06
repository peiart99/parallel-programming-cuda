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
    int k_length = ((block_size * elements_per_thread) / 2) + (blockIdx.y * block_size * elements_per_thread);
    int d_i = threadIdx.y + (blockIdx.y * block_size * (elements_per_thread / 2));
    int d_j = threadIdx.x + (blockIdx.x * block_size * (elements_per_thread / 2));
    for(int k_i {0}; k_i < elements_per_thread; k_i++)
    {
        //d_i = threadIdx.y + (blockIdx.y * block_size * elements_per_thread) + (block_size * k_i);
            //d_j = threadIdx.x + (blockIdx.x * block_size * elements_per_thread) + (block_size * k_i);
            //printf("Starting indices: [%d][%d]\n", d_i, d_j);
            for(int i {d_i}; i < 2 * reach + 1 + d_i; i++)
            {
                for(int j {d_j}; j < 2 * reach + 1 + d_j; j++)
                {
                    out_array[(d_i * (array_length - (2 * reach))) + d_j] += data_array[(i * array_length) + j];
                }
            }
            //printf("Saving into [%d][%d] from thr: [%d][%d] blck: [%d][%d] iter %d\n", d_i, d_j, threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x, k_i);

            if(k_i < ((elements_per_thread / 2) - 1))
            {
                d_i += block_size;
            }else if (k_i == ((elements_per_thread / 2) - 1))
            {
                d_i = threadIdx.y + (blockIdx.y * block_size * (elements_per_thread / 2));
                d_j += block_size;
            }else if(k_i > ((elements_per_thread / 2) - 1))
            {
                d_i += block_size;
                //d_j += block_size;
            }

//        if(d_i + block_size <= k_length)
//        {
//            d_i += block_size;
//        }else
//        {
//            d_i = threadIdx.y + (blockIdx.y * block_size * (elements_per_thread / 2));
//            if(d_j + block_size <= k_length)
//            {
//                d_j += block_size;
//            }else
//            {
//                d_j = threadIdx.x + (blockIdx.x * block_size * (elements_per_thread / 2));
//            }
//        }

    }

//    int tid = threadIdx.x * elements_per_thread;
//    int bid = threadIdx.y;
//    for(int k {0}; k < elements_per_thread; k++)
//    {
//        for(int i {bid}; i < 2 * reach + 1 + bid; i++)
//        {
//            for(int j {tid}; j < 2 * reach + 1 + tid; j++)
//            {
//                out_array[(bid * (array_length - (2 * reach))) + tid] += data_array[(i * array_length) + j];
//            }
//        }
//        tid++;
//    }

}

__global__ void deviceCalculateAnswer_test(int array_length, int reach, int elements_per_thread, float *data_array, float *out_array)
{
    int tid = threadIdx.x * elements_per_thread;
    int bid = threadIdx.y;
    for(int k {0}; k < elements_per_thread; k++)
    {
        for(int j {tid}; j < 2 * reach + 1 + tid; j++)
        {
            for(int i {bid}; i < 2 * reach + 1 + bid; i++)
            {
                out_array[(bid * (array_length - (2 * reach))) + tid] += data_array[(i * array_length) + j];
            }
        }
        tid++;
    }

}

__global__ void deviceCalculateAnswer_shared(int array_length, int reach, int elements_per_thread, int data_move_per_thread, float *data_array, float *out_array)
{
    int tid = threadIdx.x * elements_per_thread;
    int bid = threadIdx.y;
    //printf("block: %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, blockDim.z);
    extern __shared__ float shared_data[];

    int shared_index = (threadIdx.y * (array_length - (2 * reach)) * data_move_per_thread) + (threadIdx.x * data_move_per_thread);

    //printf("[%d][%d] : %d\n", threadIdx.y, threadIdx.x, shared_index);

    for(int i {shared_index}; i < (shared_index + data_move_per_thread); i++)
    {
        if(i < (array_length * array_length))
        {
            shared_data[i] = data_array[i];
            //printf("elements: %d\n", elements_per_thread);
            //printf("index: %d data: %f shared: %f\n", i, data_array[i], shared_data[i]);
        }else
        {
            //break;
        }
    }

    __syncthreads();

    for(int k {0}; k < elements_per_thread; k++)
    {
        for(int i {bid}; i < 2 * reach + 1 + bid; i++)
        {
            for(int j {tid}; j < 2 * reach + 1 + tid; j++)
            {
                out_array[(bid * (array_length - (2 * reach))) + tid] += shared_data[(i * array_length) + j];
            }
        }
        tid++;
    }

}

__global__ void testKernel(int bsize)
{
    int t_i = threadIdx.y + (blockIdx.y * bsize);
    int t_j = threadIdx.x + (blockIdx.x * bsize);
    printf("Block [%d][%d], thread [%d][%d]\n", blockIdx.y, blockIdx.x,t_i, t_j);
}