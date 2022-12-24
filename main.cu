#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

void calculateSquare(int array_length, int reach, int central_i, int central_j, float *data_array, float *out_array);

double generateRandomFloat(float min_value, float max_value)
{
    std::default_random_engine eng;
    unsigned long int t = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    eng.seed(t);
    static std::mt19937 gen(eng());
    std::uniform_real_distribution<> dist(min_value,max_value);
    return dist(gen);
}

void printArray(int array_length, float *array)
{
    for(int i {0}; i < array_length; i++)
    {

        for(int j {0}; j < array_length; j++)
        {
            std::cout << array[(array_length * i) + j] << ", ";
        }
        std::cout << std::endl;
    }
}

void populateArray(int array_length, float *array, bool zero)
{
    for(int i {0}; i < array_length; i++)
    {
        for(int j {0}; j < array_length; j++)
        {
            if(!zero)
            {
                //array[i][j] = generateRandomFloat(1.0f, 100.0f);
                array[(array_length * i) + j] = 1;
            }else
            {
                array[(array_length * i) + j] = 0;
            }

        }
    }
}

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

int main()
{
    int array_length {10};
    int reach {2};

    auto *data_array = new float[array_length * array_length];
    auto *out_array = new float[(array_length - (2 * reach)) * (array_length - (2 * reach))];
    float *device_data_array, *device_out_array;
    size_t bytes_data {(array_length * array_length) * sizeof(float)};
    size_t bytes_out {((array_length - (2 * reach)) * (array_length - (2 * reach))) * sizeof(float)};
    cudaMalloc(&device_data_array, bytes_data);
    cudaMalloc(&device_out_array, bytes_out);
    populateArray(array_length, data_array, false);
    populateArray(array_length - (2 * reach), out_array, true);
    cudaMemcpy(device_data_array, data_array, bytes_data, cudaMemcpyHostToDevice);
    printArray(array_length, data_array);
    calculateAnswer(array_length, reach, data_array, out_array);
    std::cout << "ANSWER" << std::endl;
    printArray(array_length - (2 * reach), out_array);

    deviceCalculateAnswer<<<array_length - (2 * reach), array_length - (2 * reach)>>>(array_length, reach, device_data_array, device_out_array);
    cudaDeviceSynchronize();
    cudaMemcpy(out_array, device_out_array, bytes_out, cudaMemcpyDeviceToHost);
    std::cout << "GPU ANSWER" << std::endl;
    printArray(array_length - (2 * reach), out_array);


    delete[] data_array;
    delete[] out_array;
    cudaFree(device_data_array);
    cudaFree(device_out_array);
    return 0;
}