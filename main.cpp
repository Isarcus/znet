#include "znet.hpp"

#include <iostream>
#include <random>
#include <chrono>
#include <string>

using namespace znet;

static const int INPUT_SIZE = 2;

double rand(double min, double max);
void print(dataset_t data);

int main(int argc, char** argv)
{
    Network nw(2, 2, {5, 5});

    nw.randomizeWeights();
    dataset_t input;

    input = {0, 0};
    print(nw.process(input));
    std::cout << "\n";
    for (int i = 0; i < 200000; i++)
    {
        input[0] = i % 2;
        input[1] = (i + 1) % 2;

        nw.train(input, dataset_t{input[1], input[0]}, 0.0009);
    }
    //nw.printAll();
    input = {1, 0};
    print(nw.process(input));
}

double rand(double min, double max)
{
    static bool calledYet = false;
    static std::default_random_engine eng;
    
    if (!calledYet)
    {
        eng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        calledYet = true;
    }
    std::uniform_real_distribution<double> uniform(min, max);

    return uniform(eng);
}

void print(dataset_t data)
{
    for (unsigned i = 0; i < data.size(); i++)
    {
        std::cout << " -> nn #" << i << " = " << data[i] << "\n";
    }
}