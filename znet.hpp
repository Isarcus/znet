#ifndef ZNET_HPP
#define ZNET_HPP

#include "neuron.hpp"

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <exception>

namespace znet
{
    typedef std::vector<double> dataset_t;

    class Network
    {
    public:
        Network();
        Network(int inputSize, int outputSize, std::vector<int> hiddenSizes);

        Neuron& at(int layer, int neuronID);
        const Neuron& at(int layer, int neuronID) const;

        void randomizeWeights();
        void resetNeuronInputs();
        void train(const dataset_t& input, const dataset_t& correct, double learningRate);
        dataset_t process(const dataset_t& input);

        static void assignDeltas(layer_t& layer);
        static void assignDeltas(layer_t& layerLast, const dataset_t& correct);

    private:
        std::vector<layer_t> layers;

        void addLayer(int neurons, double defaultBiasFactor = 0);
    };

    double sigmoid(const double& v, const bool& deriv);
    double identity(const double& v, const bool& deriv);

    double computeMSE(dataset_t& correct, const dataset_t& actual);

    template<typename T1, typename T2>
    void exit_if_bad_vecs(const std::vector<T1>& v1, const std::vector<T2>& v2, std::string reason);
    
}

//                          //
// TEMPLATE IMPLEMENTATIONS //
//                          //

namespace znet
{

template<typename T1, typename T2>
void exit_if_bad_vecs(const std::vector<T1>& v1, const std::vector<T2>& v2, std::string reason)
{
    if (!v1.size() || v1.size() != v2.size())
    {
        std::cerr << "Invalid sizes [" << reason << "]:\n"
        << " -> first  = " << v1.size()
        << " -> second = " << v2.size();

        exit(-1);
    }
}

}

#endif