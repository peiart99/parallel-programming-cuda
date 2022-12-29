#include <iostream>
#include "Generator.h"
#include "Array.h"
#include "Calculations.h"
#include <chrono>

void calculate(int array_length, int reach, float *data_array, float *out_array, float *device_data_array, float *device_out_array, size_t bytes_data, size_t bytes_out)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << prop.name << std::endl;
    std::cout << prop.major << "." << prop.minor << std::endl;
    std::cout << prop.multiProcessorCount << std::endl;
    std::cout << prop.maxThreadsPerBlock << std::endl;
    std::cout << prop.maxBlocksPerMultiProcessor << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //printArray(array_length, data_array);
    auto start_cpu = std::chrono::steady_clock::now();
    calculateAnswer(array_length, reach, data_array, out_array);
    auto stop_cpu = std::chrono::steady_clock::now();
    std::cout << "ANSWER CALCULATED IN " << std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count() << "us" << std::endl;
    //printArray(array_length - (2 * reach), out_array);

    // define the dimensions of the grid and thread blocks
    dim3 threads_per_block(array_length - (2 * reach),array_length - (2 * reach));
    dim3 number_of_blocks((array_length - (2 * reach)) / threads_per_block.x, (array_length - (2 * reach)) / threads_per_block.y);

    // cuda kernel call
    cudaEventRecord(start);
    deviceCalculateAnswer<<<number_of_blocks, threads_per_block>>>(array_length, reach, device_data_array, device_out_array);
    cudaEventRecord(stop);

    // wait for the device to finish executing before continuing
    cudaDeviceSynchronize();

    std::cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << '\n';
    // copy the result stored in the array in the device to a host array
    cudaMemcpy(out_array, device_out_array, bytes_out, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU ANSWER CALCULATED IN: " << milliseconds * 1000.0f << " us" << std::endl;
    //printArray(array_length - (2 * reach), out_array);
}

int main()
{
    int array_length {10};
    int reach {2};
    auto *data_array = new float[array_length * array_length];
    auto *out_array = new float[(array_length - (2 * reach)) * (array_length - (2 * reach))];
    float *device_data_array, *device_out_array;

    // calculate the size in bytes of host arrays
    size_t bytes_data {(array_length * array_length) * sizeof(float)};
    size_t bytes_out {((array_length - (2 * reach)) * (array_length - (2 * reach))) * sizeof(float)};

    // allocate memory on the device equal to host data that's going to be used by the device
    cudaMalloc(&device_data_array, bytes_data);
    cudaMalloc(&device_out_array, bytes_out);

    // fill the data array (setting the last parameter named "debug" to "true" fills the array with 1's to make debugging easy)
    populateArray(array_length, data_array, false);

    // copy the host data array's content into the memory previously allocated on the device
    cudaMemcpy(device_data_array, data_array, bytes_data, cudaMemcpyHostToDevice);

    calculate(array_length, reach, data_array, out_array, device_data_array, device_out_array, bytes_data, bytes_out);

    // free the heap allocated to dynamically created host arrays
    delete[] data_array;
    delete[] out_array;

    // free the memory allocated to arrays on the device
    cudaFree(device_data_array);
    cudaFree(device_out_array);
    return 0;
}