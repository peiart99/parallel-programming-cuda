#include "Generator.h"

double generateRandomFloat(float min_value, float max_value)
{
    std::default_random_engine eng;
    unsigned long int t = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    eng.seed(t);
    static std::mt19937 gen(eng());
    std::uniform_real_distribution<> dist(min_value,max_value);
    return dist(gen);
}