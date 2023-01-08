#include <iostream>
#include "Generator.h"
#include "Array.h"
#include "Calculations.h"
#include <chrono>

void setParameters(int &data_array_length, int &reach, int &block_size, int &elements_per_thread)
{
    int new_value {0};
    do {
        std::cout << "Podaj wartość parametru N" << std::endl;
        std::cin >> new_value;
    }while(new_value < 10);
    data_array_length = new_value;
    new_value = 0;
    do {
        std::cout << "Podaj wartość parametru R" << std::endl;
        std::cin >> new_value;
    }while(new_value < 1);
    reach = new_value;
    new_value = 0;
    do {
        std::cout << "Podaj wartość parametru BS(8, 16, 32)" << std::endl;
        std::cin >> new_value;
    }while(new_value != 8 && new_value != 16 && new_value != 32);
    block_size = new_value;
    new_value = 0;
    do {
        std::cout << "Podaj wartość parametru K" << std::endl;
        std::cin >> new_value;
    }while(new_value < 1);
    elements_per_thread = new_value;
    new_value = 0;

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

void calculate(bool use_shared_memory)
{

    // Przypisanie wartości parametrów początkowych ========================================================================================
    int data_array_length {};
    int reach {};
    int block_size {};
    int elements_per_thread {};
    setParameters(data_array_length, reach, block_size, elements_per_thread);
    int out_array_length {data_array_length - (2 * reach)};
    int elements_per_thread_modifier {};
    elements_per_thread_modifier = (elements_per_thread > 1) ? (elements_per_thread_modifier = elements_per_thread/2) : elements_per_thread_modifier = 1;
    int shared_memory_length {(elements_per_thread_modifier * block_size) + (2 * reach)};
    unsigned int shared_memory_size_in_bytes = shared_memory_length * shared_memory_length * sizeof(float);
    double shared_elements_per_thread {std::ceil(static_cast<double>(shared_memory_length * shared_memory_length) / (block_size * block_size))};
    auto *data_array = new float[data_array_length * data_array_length];
    auto *out_array = new float[out_array_length * out_array_length];
    auto *out_array_device_global_mem = new float[out_array_length * out_array_length];
    float *device_data_array, *device_out_array;
    dim3 threads_per_block(block_size,block_size);
    dim3 number_of_blocks(((out_array_length) / threads_per_block.x) / elements_per_thread_modifier, ((out_array_length) / threads_per_block.y) / elements_per_thread_modifier);
    size_t bytes_data {(data_array_length * data_array_length) * sizeof(float)};
    size_t bytes_out {(out_array_length * out_array_length) * sizeof(float)};
    std::string message{};
    message = (use_shared_memory) ? message = "współdzielonej": message = "globalnej";
    // =====================================================================================================================================


    // Alokacja miejsca w pamięci GPU odpowiadająca rozmiarom tablic wejściowej oraz wyjściowej
    cudaMalloc(&device_data_array, bytes_data);
    cudaMalloc(&device_out_array, bytes_out);

    // Wypełnienie tablicy wejściowej danymi - ustawienie parametru "debug" na "true" wypełnia tablicę jedynkami - przydatne do debugowania
    populateArray(data_array_length, data_array, false);

    // Synchroniczne kopiowanie danych z pamięci hosta do zarezerwowanego wcześniej miejsca w pamięci GPU
    cudaMemcpy(device_data_array, data_array, bytes_data, cudaMemcpyHostToDevice);

    // Ustawienie cudaEvent użytego do pomiaru czasu wykonywania obliczeń na GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printDeviceInfo(0);

    std::cout << "N: " << data_array_length << " Wielkość tabeli wejściowej: " << bytes_data / 1000 << " kB" << std::endl;
    if(use_shared_memory)
    {
        std::cout << "PW: " << shared_memory_length << " Wielkość pamięci współdzielonej: " << shared_memory_size_in_bytes / 1000 << " kB" << std::endl;
    }
    std::cout << "R: " << reach << std::endl;
    std::cout << "BS: " << block_size << "x" << block_size << std::endl;
    std::cout << "K: " << elements_per_thread << std::endl;

    // Obliczenie rozwiązania sekwencyjnie na CPU wraz z pomiarem czasu wykonywania
    auto start_cpu = std::chrono::steady_clock::now();
    calculateAnswer(data_array_length, reach, data_array, out_array);
    auto stop_cpu = std::chrono::steady_clock::now();
    std::cout << "Czas obliczeń dla rozwiązania sekwencyjnego (CPU): " << std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count() << " us" << std::endl;

    // Rozpoczęcie pomiaru czasu dla GPU
    cudaEventRecord(start);

    // Wywołanie odpowiedniego kernela (dla pamięci globalnej lub współdzielonej) z odpowiednimi parametrami
    if(use_shared_memory)
    {
        deviceCalculateAnswer_shared<<<number_of_blocks, threads_per_block, shared_memory_size_in_bytes>>>(data_array_length, reach, elements_per_thread, block_size, shared_memory_length ,static_cast<int>(shared_elements_per_thread) , device_data_array, device_out_array);
    }else
    {
        deviceCalculateAnswer<<<number_of_blocks, threads_per_block>>>(data_array_length, reach, elements_per_thread, block_size, device_data_array, device_out_array);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // "blokada" wykonywania kodu hosta aż do momentu kiedy GPU skończy obliczenia
    cudaDeviceSynchronize();

    std::cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << '\n';

    // Synchroniczne kopiowanie wyniku z pamięci GPU do tablicy w pamięci hosta
    cudaMemcpy(out_array_device_global_mem, device_out_array, bytes_out, cudaMemcpyDeviceToHost);

    // Porównanie wyników sekwencyjnego(CPU) z wynikiem obliczeń GPU
    int conflicts = compareArrays(out_array_length, out_array, out_array_device_global_mem);
    std::cout << "Konflikty (Wynik CPU/GPU): " << conflicts << std::endl;
    std::cout << "Czas obliczeń dla rozwiązania z użyciem pamięci " << message <<  " (GPU): " << milliseconds * 1000.0f << " us" << std::endl;
    float percentage = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count() / (milliseconds * 1000.0f);
    std::cout << "Obliczenia na GPU wykonane " << percentage << " razy szybciej niż rozwiązanie sekwencyjne CPU" << std::endl;

    // zwolnienie dynamicznie przydzielonej pamięci
    delete[] data_array;
    delete[] out_array;

    // zwolnienie pamięci przydzielonej na pamięci GPU
    cudaFree(device_data_array);
    cudaFree(device_out_array);
}

int main()
{
    bool choice_made {false};
    bool use_shared_memory {false};
    int choice {};
    do {
        std::cout << "[1] Użycie pamięci globalnej GPU\n[2] Użycie pamięci współdzielonej GPU" << std::endl;
        std::cin >> choice;
        switch(choice) {
            case 1:
            {
                use_shared_memory = false;
                choice_made = true;
                break;
            }
            case 2:
            {
                use_shared_memory = true;
                choice_made = true;
                break;
            }
            default:
            {
                std::cout << "Niepoprawny wybór" << std::endl;
            }
        }
    }while(!choice_made);
    calculate(use_shared_memory);
    cudaDeviceReset();
    return 0;
}