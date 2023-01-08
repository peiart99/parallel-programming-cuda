#include <iostream>
#include "Generator.h"
#include "Array.h"
#include "Calculations.h"
#include <chrono>

void setParameters(int data_array_length, int reach, int block_size, int elements_per_thread)
{
    //
}

void printDeviceInfo(int device_number)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_number);
    std::cout << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Multiprocesory (SM): " <<prop.multiProcessorCount << std::endl;
    std::cout << "Maksymalna liczba wątków na blok: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Maksymalna liczba bloków na SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
}

void calculate(int array_length, int reach, int elements_per_thread, float *data_array, float *out_array, float *out_array_device_global_mem, float *device_data_array, float *device_out_array, size_t bytes_data, size_t bytes_out)
{
    printDeviceInfo(0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //printArray(array_length, data_array);
    std::cout << "DATA ARRAY SIZE: " << ((array_length * array_length) * sizeof(float)) / 1000 << " kB"<< std::endl;
    auto start_cpu = std::chrono::steady_clock::now();
    calculateAnswer(array_length, reach, data_array, out_array);
    auto stop_cpu = std::chrono::steady_clock::now();

    std::cout << "ANSWER CALCULATED IN " << std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count() << "us" << std::endl;
    //printArray(array_length - (2 * reach), out_array);
    // define the dimensions of the grid and thread blocks
    int BS {8};
    int divisor {1};
    divisor = (elements_per_thread > 1) ? (divisor = elements_per_thread/2) : divisor = 1;
    double data_move_per_thread {std::ceil(static_cast<double>(((divisor * BS) + (2 * reach)) * ((divisor * BS) + (2 * reach))) / (BS * BS))};
    std::cout << "DATA PER THREAD MOVE: " << data_move_per_thread << std::endl;
    dim3 threads_per_block(BS,BS);
    dim3 number_of_blocks(((array_length - (2 * reach)) / threads_per_block.x) / divisor, ((array_length - (2 * reach)) / threads_per_block.y) / divisor);

    dim3 lol(8,8);
    dim3 lel(2,2);

    std::cout << "Blocks: " << number_of_blocks.x << ", " << number_of_blocks.y << std::endl;
    std::cout << "Shared size: " << (((divisor *  BS) + (2 * reach)) * ((divisor * BS) + (2 * reach)) * sizeof(float)) / 1000 << " kB" << std::endl;

    // cuda kernel call
    cudaEventRecord(start);

    deviceCalculateAnswer<<<number_of_blocks, threads_per_block>>>(array_length, reach, elements_per_thread, BS, device_data_array, device_out_array);
    //deviceCalculateAnswer_shared<<<number_of_blocks, threads_per_block, (((divisor * BS) + (2 * reach)) * ((divisor * BS) + (2 * reach)) * sizeof(float))>>>(array_length, reach, elements_per_thread, BS, ((divisor * BS) + (2 * reach)),static_cast<int>(data_move_per_thread) , device_data_array, device_out_array);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // wait for the device to finish executing before continuing
    cudaDeviceSynchronize();

    std::cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << '\n';
    // copy the result stored in the array in the device to a host array
    cudaMemcpy(out_array_device_global_mem, device_out_array, bytes_out, cudaMemcpyDeviceToHost);
    std::cout << "GPU ANSWER CALCULATED IN: " << milliseconds * 1000.0f << " us" << std::endl;

    float percentage = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count() / (milliseconds * 1000.0f);
    std::cout << "GPU is " << percentage << " times faster than the CPU" << std::endl;
    //printArray(array_length - (2 * reach), out_array_device_global_mem);
    int conflicts = compareArrays(array_length - (2 * reach), out_array, out_array_device_global_mem);
    std::cout << "Conflicts: " << conflicts << std::endl;

}

int main()
{
    int array_length {156};
    int reach {46};
    int elements_per_thread {4};
    auto *data_array = new float[array_length * array_length];
    auto *out_array = new float[(array_length - (2 * reach)) * (array_length - (2 * reach))];
    auto *out_array_device_global_mem = new float[(array_length - (2 * reach)) * (array_length - (2 * reach))];
    float *device_data_array, *device_out_array;
    //double data_move_per_thread {std::ceil(static_cast<double>(array_length * array_length) / ((array_length - (2 * reach)) * (array_length - (2 * reach))))};

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

    calculate(array_length, reach, elements_per_thread, data_array, out_array, out_array_device_global_mem, device_data_array, device_out_array, bytes_data, bytes_out);

    // free the heap allocated to dynamically created host arrays
    delete[] data_array;
    delete[] out_array;

    // free the memory allocated to arrays on the device
    cudaFree(device_data_array);
    cudaFree(device_out_array);

    cudaDeviceReset();
    return 0;
}