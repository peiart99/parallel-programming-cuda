#include "Array.h"

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

void populateArray(int array_length, float *array, bool debug)
{
    for(int i {0}; i < array_length; i++)
    {
        for(int j {0}; j < array_length; j++)
        {
            if(!debug)
            {
                array[(array_length * i) + j] = generateRandomFloat(1.0f, 100.0f);
            }else
            {
                array[(array_length * i) + j] = 1;
            }

        }
    }
}

int compareArrays(int array_length, float *a, float *b)
{
    int conflicts {0};
    for(int i {0}; i < array_length * array_length; i++)
    {
        if(a[i] != b[i])
        {
            conflicts++;
        }
    }

    return conflicts;
}