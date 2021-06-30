#include "load_mnist.h"

#include <fstream>
#include <iostream>
#include <iomanip>

namespace znet
{

//       //
// Image //
//       //

const byte& Image::at(int x, int y) const
{
    return data[x * MNIST_IMG_WIDTH + y];
}

byte& Image::at(int x, int y)
{
    return data[x * MNIST_IMG_WIDTH + y];
}

void Image::print() const
{
    std::cout << "Label " << label << "\n";
    for (int x = 0; x < MNIST_IMG_WIDTH; x++)
    {
        for (int y = 0; y < MNIST_IMG_HEIGHT; y++)
        {
            std::cout << std::setw(4) << (int)at(x, y);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void Image::paste(std::vector<double>& into) const
{
    if (into.size() < MNIST_IMG_SIZE)
    {
        throw std::runtime_error("Incorrect vector size!");
    }

    for (int i = 0; i < MNIST_IMG_SIZE; i++)
    {
        into[i] = data[i] / 255.0;
    }
}

//          //
// ImageSet //
//          //

ImageSet::ImageSet(std::string path_images, std::string path_labels)
    : idx(0)
{
    std::ifstream fin_image(path_images, std::ios_base::binary);
    if (fin_image.fail())
    {
        throw std::runtime_error("Could not load training data at " + path_images);
    }

    std::ifstream fin_label(path_labels, std::ios_base::binary);
    if (fin_label.fail())
    {
        throw std::runtime_error("Could not load training data at " + path_labels);
    }

    // Load header from images
    char header_image[16];
    fin_image.read(header_image, 16);

    // Load header from labels
    char header_label[8];
    fin_label.read(header_label, 8);

    // Read in all images
    while (true)
    {
        Image* img = new Image;
        fin_image.read((char*)img->data, MNIST_IMG_SIZE);

        // Exit loop if reading image failed
        if (fin_image.fail())
        {
            delete img;
            break;
        }

        // Load image label
        char label;
        fin_label.read(&label, 1);
        img->label = label;

        images.push_back(img);
    }

    std::cout << "Loaded " << images.size() << " training images\n";
}

ImageSet::~ImageSet()
{
    for (Image*& img : images)
    {
        delete img;
    }
}

const Image* ImageSet::at(int i) const
{
    return images.at(i);
}

const Image* ImageSet::nextImage()
{
    if (idx == images.size())
    {
        return nullptr;
    }

    return images[idx++];
}

trainingset_t* ImageSet::convertToRaw() const
{
    trainingset_t* data = new trainingset_t;
    data->inputs = std::vector<dataset_t>(images.size());
    data->outputs = std::vector<dataset_t>(images.size());

    dataset_t img_data(MNIST_IMG_SIZE);
    for (unsigned i = 0; i < images.size(); i++)
    {
        images[i]->paste(img_data);
        data->inputs.at(i) = img_data;
        data->outputs.at(i) = dataset_t(10);
        data->outputs[i][images[i]->label] = 1;
    }

    return data;
}

} // namespace znet