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
            //std::cout << "SEKW: Dla OUT[" << central_i << "][" << central_j << "] wartosci: " << data_array[(i * array_length) + j]<< std::endl;
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
            //printf("Saving into [%d][%d] from thr: [%d][%d] blck: [%d][%d] iter %d\n", d_i, d_j, threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x, k_i);

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

    int shared_start_i = blockIdx.y * block_size;
    int shared_start_j = blockIdx.x * block_size;
    int shared_end_i = shared_start_i + (shared_size - 1);
    int shared_end_j = shared_start_j + (shared_size - 1);

    int t_id = (block_size * threadIdx.y) + threadIdx.x;
    int s_i = (d_i * block_size) + d_j;

    int elements_counted = t_id * data_move_per_thread;

    int thread_shared_start_i = (elements_counted / shared_size);
    int thread_shared_start_j = (elements_counted % shared_size);

    int counter {0};


    extern __shared__ float shared_data[];

    //for(int i {d_i}; i < )



    for(int i {thread_shared_start_i}; i < shared_size; i++)
    {
        for(int j {thread_shared_start_j}; j < shared_size; j++)
        {
            if(counter == data_move_per_thread)
            {
                goto endloop;
            }
            shared_data[(i * shared_size) + j] = data_array[((i + shared_start_i) * array_length) + (j + shared_start_j)];
            //printf("block[%d][%d] = shared[%d][%d] = data[%d][%d]\n",blockIdx.y,blockIdx.x,i,j,(i + shared_start_i), (j + shared_start_j));

            counter++;
        }
        thread_shared_start_j = 0;
    }
    endloop:

    __syncthreads();

    int counterlol {0};



    for(int k_i {0}; k_i < elements_per_thread; k_i++)
    {
        for(int i = threadIdx.y; i < (2 * reach) + 1 +threadIdx.y; i++)
        {
            for(int j = threadIdx.x; j < (2 * reach) + 1 + threadIdx.x; j++)
            {
                out_array[(d_i * (array_length - (2 * reach))) + d_j] += shared_data[(i * shared_size) + j];
                counterlol++;
                //printf("SHARED: Dla OUT[%d][%d] wartosci indeks[%d][%d]: %f \n", d_i, d_j,i,j, shared_data[(i * shared_size) + j]);
            }
        }

        //printf("byku powinno byc %d a masz %d xd \n", (((2*reach) + 1) * ((2*reach) + 1)), counterlol);
        //printf("Saving into [%d][%d] from thr: [%d][%d] blck: [%d][%d] iter %d\n", d_i, d_j, threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x, k_i);

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

    __syncthreads();








//    int tid = threadIdx.x * elements_per_thread;
//    int bid = threadIdx.y;
//    //printf("block: %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, blockDim.z);
//    extern __shared__ float shared_data[];
//
//    int shared_index = (threadIdx.y * (array_length - (2 * reach)) * data_move_per_thread) + (threadIdx.x * data_move_per_thread);
//
//    //printf("[%d][%d] : %d\n", threadIdx.y, threadIdx.x, shared_index);
//
//    for(int i {shared_index}; i < (shared_index + data_move_per_thread); i++)
//    {
//        if(i < (array_length * array_length))
//        {
//            shared_data[i] = data_array[i];
//            //printf("elements: %d\n", elements_per_thread);
//            //printf("index: %d data: %f shared: %f\n", i, data_array[i], shared_data[i]);
//        }else
//        {
//            //break;
//        }
//    }
//
//    __syncthreads();
//
//    for(int k {0}; k < elements_per_thread; k++)
//    {
//        for(int i {bid}; i < 2 * reach + 1 + bid; i++)
//        {
//            for(int j {tid}; j < 2 * reach + 1 + tid; j++)
//            {
//                out_array[(bid * (array_length - (2 * reach))) + tid] += shared_data[(i * array_length) + j];
//            }
//        }
//        tid++;
//    }

}