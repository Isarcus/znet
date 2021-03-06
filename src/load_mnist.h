#ifndef LOAD_MNIST_H
#define LOAD_MNIST_H

#include "znet.h"

#include <string>
#include <vector>

namespace znet
{
    static const int MNIST_IMG_WIDTH = 28;
    static const int MNIST_IMG_HEIGHT = 28;
    static const int MNIST_IMG_SIZE = MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT;

    typedef unsigned char byte;

    typedef struct BrightnessMap
    {
        int label;
        byte data[MNIST_IMG_SIZE];

        const byte& at(int x, int y) const;
        byte& at(int x, int y);
        void print() const;

        void paste(std::vector<double>& into) const;
    } BrightnessMap;

    class ImageSet
    {
    public:
        ImageSet(std::string path_images, std::string path_labels);
        ~ImageSet();

        const BrightnessMap* at(int i) const;
        const BrightnessMap* nextImage();

        trainingset_t* convertToRaw() const;

    private:
        unsigned idx;
        std::vector<BrightnessMap*> images;
    };

}

#endif