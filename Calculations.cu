#include "Calculations.h"

void calculateAnswer(int data_array_length, int reach, float *data_array, float *out_array)
{
    for(int i {0}; i < (data_array_length - (2 * reach)); i++)
    {
        for(int j {0}; j < (data_array_length - (2 * reach)); j++)
        {
            calculateSquare(data_array_length, reach, i, j, data_array, out_array);
        }
    }
}


void calculateSquare(int data_array_length, int reach, int central_i, int central_j, float *data_array, float *out_array)
{
    for(int i {central_i}; i < 2 * reach + 1 + central_i; i++)
    {
        for(int j {central_j}; j < 2 * reach + 1 + central_j; j++)
        {
            out_array[(central_i * (data_array_length - (2 * reach))) + central_j] += data_array[(i * data_array_length) + j];
        }
    }
}

__global__ void deviceCalculateAnswer(int data_array_length, int reach, int elements_per_thread, int block_size, float *data_array, float *out_array)
{

    int elements_per_thread_modifier {1};
    if(elements_per_thread > 1)
    {
        elements_per_thread_modifier = elements_per_thread / 2;
    }

    // indeksy wyznaczające obliczany element tablicy wyjściowej - poprawne niezależnie od liczby bloków czy parametru K
    int global_thread_i = threadIdx.y + (blockIdx.y * block_size * elements_per_thread_modifier);
    int global_thread_j = threadIdx.x + (blockIdx.x * block_size * elements_per_thread_modifier);

    for(int k_i {0}; k_i < elements_per_thread; k_i++)
    {
            for(int i {global_thread_i}; i < 2 * reach + 1 + global_thread_i; i++)
            {
                for(int j {global_thread_j}; j < 2 * reach + 1 + global_thread_j; j++)
                {
                    out_array[(global_thread_i * (data_array_length - (2 * reach))) + global_thread_j] += data_array[(i * data_array_length) + j];
                }
            }

            // Kod obsługujące poprawne wykonanie dla parametru K > 1. Każdy blok wykonuje pracę K bloków dlatego konieczne są odpowiednie przesunięcia indeksów
            if(k_i < (elements_per_thread_modifier - 1))
            {
                global_thread_i += block_size;
            }else if (k_i == (elements_per_thread_modifier - 1))
            {
                global_thread_i = threadIdx.y + (blockIdx.y * block_size * elements_per_thread_modifier);
                global_thread_j += block_size;
            }else if(k_i > (elements_per_thread_modifier - 1))
            {
                global_thread_i += block_size;
            }
    }
}

__global__ void deviceCalculateAnswer_shared(int data_array_length, int reach, int elements_per_thread, int block_size, int shared_size, int shared_elements_per_thread, float *data_array, float *out_array)
{

    int k_multiplier {1};
    if(elements_per_thread > 1)
    {
        k_multiplier = elements_per_thread / 2;
    }

    // indeksy wyznaczające obliczany element tablicy wyjściowej - poprawne niezależnie od liczby bloków czy parametru K
    int global_thread_i = threadIdx.y + (blockIdx.y * block_size * k_multiplier);
    int global_thread_j = threadIdx.x + (blockIdx.x * block_size * k_multiplier);

    // modyfikatory pomocnicze dla przesunięcia indeksów pamięci współdzielonej dla K > 1
    int k_shmem_modifier_i {0};
    int k_shmem_modifier_j {0};

    // modyfikatory pomocnicze dla przesunięcia indeksów tablicy wejściowej podczas wczytywania danych do pamięci współdzielonej przy wielu blokach
    int shared_start_i = blockIdx.y * (block_size * k_multiplier);
    int shared_start_j = blockIdx.x * (block_size * k_multiplier);

    // lokalny jednowymiarowy indeks wątku
    int thread_local_id = (block_size * threadIdx.y) + threadIdx.x;

    // liczba elementów pamięci współdzielonej obsłużonej przez wątki z ID mniejszym niż obecny wątek
    int elements_counted = thread_local_id * shared_elements_per_thread;

    // działania wyznaczające indeksy od których obecny wątek kontynuuje przenoszenie danych z tablicy wejściowej do pamięci współdzielonej
    int thread_shared_start_i = (elements_counted / shared_size);
    int thread_shared_start_j = (elements_counted % shared_size);

    int counter {0};

    // deklaracja tablicy w pamięci współdzielonej
    extern __shared__ float shared_data[];

    /* pętle obsługujące załadowanie danych tablicy wejściowej do pamięci współdzielonej
     * w celu zminimalizowania używanego obszaru pamięci wspóldzielonej (której limit to 48 kB)
     * każdy blok wczytuje wyłącznie dane które wczytane będą w jego zasięgu, czyli BS + 2R z odpowiednim powiększeniem
     * o K (dla K=4 jest to 2BS + 2R). Innym możliwym podejściem jest wczytywanie obszaru BS + 2R dla KAŻDEJ iteracji K
     * w danym bloku, jednakże zyski w obszarze zużycia pamięci nie wydają się warte zwielokrotnienia czasu
     * ładowania danych do pamięci współzielonej o K.
     * */
    for(int i {thread_shared_start_i}; i < shared_size; i++)
    {
        for(int j {thread_shared_start_j}; j < shared_size; j++)
        {
            if(counter == shared_elements_per_thread)
            {
                goto endloop;
            }
            shared_data[(i * shared_size) + j] = data_array[((i + shared_start_i) * data_array_length) + (j + shared_start_j)];
            counter++;
        }
        thread_shared_start_j = 0;
    }
    endloop:

    // blokada czekająca aż każdy wątek wykona kod powyżej - zastosowana w celu upewnienia się, że pamięć
    // współdzielona została zapełniona przed wykonywaniem dalszych obliczeń
    __syncthreads();


    for(int k_i {0}; k_i < elements_per_thread; k_i++)
    {
        for(int i = threadIdx.y + k_shmem_modifier_i; i < (2 * reach) + 1 + (threadIdx.y + k_shmem_modifier_i); i++)
        {
            for(int j = (threadIdx.x + k_shmem_modifier_j); j < (2 * reach) + 1 + (threadIdx.x + k_shmem_modifier_j); j++)
            {
                out_array[(global_thread_i * (data_array_length - (2 * reach))) + global_thread_j] += shared_data[(i * shared_size) + j];
            }
        }

        // Kod obsługujące poprawne wykonanie dla parametru K > 1. Każdy blok wykonuje pracę K bloków dlatego konieczne są odpowiednie przesunięcia indeksów
        // zarówno dla tablicy wejściowej jak i pamięci współdzielonej powiększonej o K
        if(k_i < (k_multiplier - 1))
        {
            global_thread_i += block_size;
            k_shmem_modifier_i += block_size;
        }else if (k_i == (k_multiplier - 1))
        {
            global_thread_i = threadIdx.y + (blockIdx.y * block_size * k_multiplier);
            k_shmem_modifier_i = 0;
            global_thread_j += block_size;
            k_shmem_modifier_j += block_size;
        }else if(k_i > (k_multiplier - 1))
        {
            global_thread_i += block_size;
            k_shmem_modifier_i += block_size;
        }
    }

    __syncthreads();

}