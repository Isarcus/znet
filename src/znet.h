#ifndef ZNET_H
#define ZNET_H

#include "neuron.h"

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <exception>

namespace znet
{
    typedef std::vector<double> dataset_t;

    typedef struct trainingset_t
    {
        std::vector<dataset_t> inputs;
        std::vector<dataset_t> outputs;
    } trainingset_t;

    class Network
    {
    public:
        Network();
        Network(int inputSize, int outputSize, std::vector<int> hiddenSizes);

        Neuron& at(int layer, int neuronID);
        const Neuron& at(int layer, int neuronID) const;

        void randomizeWeights(double range);
        void resetNeuronInputs();
        void setLayerActivationFunc(int layerIdx, activationFunc_t func);
        void train(const dataset_t& input, const dataset_t& correct, double learningRate);
        void train(const trainingset_t& batch, double learningRate);
        void train(const trainingset_t& data, int batchSize, int epochs, double learningRate);

        dataset_t getIdealInput(const dataset_t& outputCorrect, int iterations, double learningRate);

        void computeChanges(const trainingset_t& data, const int& startIdx, const int& howMany, std::vector<std::vector<dataset_t>>& changeVec);
        void applyChanges(std::vector<std::vector<dataset_t>>& changeVec, const int& batchSize, const double& learningRate);

        dataset_t process(const dataset_t &input);

        void printAll() const;

        void writeText(std::string path) const;
        void loadText(std::string path);

        static void assignDeltas(layer_t& layer);
        static void assignDeltas(layer_t& layerLast, const dataset_t& correct);

    private:
        std::vector<layer_t> layers;

        void addLayer(int neurons, double defaultBiasFactor = 0);
    };

    double sigmoid(const double& v, const bool& deriv);
    double identity(const double& v, const bool& deriv);
    double rectified(const double& v, const bool& deriv);
    double rectifiedLeaky(const double& v, const bool& deriv);

    double computeMSE(const dataset_t& correct, const dataset_t& actual);

    std::ostream& operator<<(std::ostream& os, const std::vector<std::pair<Neuron*, double>>& weights);

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
        std::cerr << "Invalid vectors [" << reason << "]:\n"
        << " -> first  = " << v1.size() << "\n"
        << " -> second = " << v2.size() << "\n";

        exit(-1);
    }
}

}

#endif