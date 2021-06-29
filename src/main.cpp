#include "znet.h"
#include "load_mnist.h"

#include <iostream>
#include <random>
#include <chrono>
#include <string>

using namespace znet;

constexpr const char* PATH_TRAIN_IMAGES = "data/train-images.idx3-ubyte";
constexpr const char* PATH_TRAIN_LABELS = "data/train-labels.idx1-ubyte";

double rand(double min, double max);
void print(dataset_t data);

int main(int argc, char** argv)
{
    ImageSet training(PATH_TRAIN_IMAGES, PATH_TRAIN_LABELS);

    Network nw(MNIST_IMG_SIZE, 10, {100, 100, 100});
    nw.randomizeWeights(0.1);

    // Train on data
    for (int i = 0; i < 2000; i++)
    {
        // Create batch
        constexpr const int SIZE_BATCH = 20;
        std::vector<dataset_t> inputBatch(SIZE_BATCH);
        std::vector<dataset_t> correctBatch(SIZE_BATCH);
        for (int j = 0; j < SIZE_BATCH; j++)
        {
            const Image* img = training.nextImage();
            inputBatch[j] = dataset_t(MNIST_IMG_SIZE);
            img->paste(inputBatch[j]);
            correctBatch[j] = dataset_t(10);
            correctBatch[j][img->label] = 1;
        }

        nw.train(inputBatch, correctBatch, 0.01);
    }

    // Test accuracy
    int numCorrect = 0;
    dataset_t input(MNIST_IMG_SIZE);
    for (int i = 0; i < 1000; i++)
    {
        const Image* img = training.nextImage();
        img->paste(input);

        // Find guess
        dataset_t output = nw.process(input);
        int guessIdx = -1;
        double guessAct = 0;
        for (int j = 0; j < 10; j++)
        {
            if (guessIdx == -1 || output[j] > guessAct)
            {
                guessIdx = j;
                guessAct = output[j];
            }
        }

        // Print guess info
        std::cout << "[" << img->label << "]: " << guessIdx << " (" << guessAct << ")\n";

        if (guessIdx == img->label) numCorrect++;
    }

    std::cout << "ACCURACY: " << numCorrect / 1000.0 << "\n";
}

//         //
// Helpers //
//         //

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