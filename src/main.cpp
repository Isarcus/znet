#include "znet.h"
#include "load_mnist.h"

#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>

using namespace znet;

constexpr const char* PATH_TRAIN_IMAGES = "data/train-images-idx3-ubyte";
constexpr const char* PATH_TRAIN_LABELS = "data/train-labels-idx1-ubyte";
constexpr const char* PATH_TEST_IMAGES = "data/test-images-idx3-ubyte";
constexpr const char* PATH_TEST_LABELS = "data/test-labels-idx1-ubyte";

double rand(double min, double max);
void print(dataset_t data);

void trainBatches();

int main(int argc, char** argv)
{
    // Network nw_train(MNIST_IMG_SIZE, 10, {100, 100, 100});
    // nw_train.randomizeWeights(0.1);

    // ImageSet* training = new ImageSet(PATH_TRAIN_IMAGES, PATH_TRAIN_LABELS);
    // auto data_train = training->convertToRaw();
    // delete training;

    // // Train
    // nw_train.train(*data_train, 30, 3, 0.1);

    // // Save configuration
    // nw_train.writeText("saves/data.txt");

    // Load configuration
    Network nw_test;
    nw_test.loadText("saves/data.txt");

    // Test accuracy on MNIST data
    ImageSet testing(PATH_TEST_IMAGES, PATH_TEST_LABELS);
    trainingset_t* testing_set = testing.convertToRaw();
    nw_test.test(*testing_set, true);

    // Test accuracy on homemade data
    ImageSet homemade("data/images_me_10x18.bt", "data/labels_me_10x18.bt");
    trainingset_t* homemade_set = homemade.convertToRaw();
    nw_test.test(*homemade_set, true);

    // Example of getting ideal input for a given output
    dataset_t outputCorrect(10);
    outputCorrect[1] = 1.0;
    dataset_t ideal = nw_test.getIdealInput(outputCorrect, 40000, 0.01);

    return 0;
}

// Train on most of the data set
void trainBatches()
{
    ImageSet training(PATH_TRAIN_IMAGES, PATH_TRAIN_LABELS);

    Network nw(MNIST_IMG_SIZE, 10, {100, 100, 100});
    nw.randomizeWeights(0.1);

    // Train on data
    for (int i = 0; i < 2000; i++)
    {
        // Create batch
        constexpr const int SIZE_BATCH = 20;
        trainingset_t data;
        data.inputs  = std::vector<dataset_t>(SIZE_BATCH);
        data.outputs = std::vector<dataset_t>(SIZE_BATCH);
        for (int j = 0; j < SIZE_BATCH; j++)
        {
            const BrightnessMap* img = training.nextImage();
            data.inputs[j] = dataset_t(MNIST_IMG_SIZE);
            img->paste(data.inputs[j]);
            data.outputs[j] = dataset_t(10);
            data.outputs[j][img->label] = 1;
        }

        nw.train(data, 0.01);

        if (!(i % 100))
        {
            std::cout << " -> " << i / 20.0 << "% done\n";
        }
    }

    // Test accuracy
    int numCorrect = 0;
    dataset_t input(MNIST_IMG_SIZE);
    for (int i = 0; i < 1000; i++)
    {
        const BrightnessMap* img = training.nextImage();
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