#include <iostream>
#include "Generator.h"
#include "Array.h"
#include "Calculations.cuh"

void calculate(int array_length, int reach, float *data_array, float *out_array, float *device_data_array, float *device_out_array, size_t bytes_data, size_t bytes_out)
{
    printArray(array_length, data_array);
    calculateAnswer(array_length, reach, data_array, out_array);
    std::cout << "ANSWER" << std::endl;
    printArray(array_length - (2 * reach), out_array);

    // cuda kernel call
    deviceCalculateAnswer<<<array_length - (2 * reach), array_length - (2 * reach)>>>(array_length, reach, device_data_array, device_out_array);

    // wait for the device to finish executing before continuing
    cudaDeviceSynchronize();

    std::cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << '\n';
    // copy the result stored in the array in the device to a host array
    cudaMemcpy(out_array, device_out_array, bytes_out, cudaMemcpyDeviceToHost);
    std::cout << "GPU ANSWER" << std::endl;
    printArray(array_length - (2 * reach), out_array);
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
    populateArray(array_length, data_array, true);

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