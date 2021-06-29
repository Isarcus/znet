#include "znet.h"

#include <cmath>
#include <random>
#include <chrono>

#define LOOP_NETWORK for (auto& layer : layers) for (auto& nn : layer)

namespace znet
{

//         //
// NETWORK //
//         //

Network::Network() {}

Network::Network(int inputSize, int outputSize, std::vector<int> hiddenSizes)
{
    addLayer(inputSize, 0);
    for (const int& size : hiddenSizes)
    {
        addLayer(size, 0);
    }
    addLayer(outputSize, 0);

    setLayerActivationFunc(0, identity);
}

Neuron& Network::at(int layer, int neuronID)
{
    return layers.at(layer).at(neuronID);
}

const Neuron& Network::at(int layer, int neuronID) const
{
    return layers.at(layer).at(neuronID);
}

// Randomize all weights for all connections to be between 0 and 1
void Network::randomizeWeights(double range)
{
    std::uniform_real_distribution<> uniform(-range, range);
    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "Randomizing with seed: " << seed << "\n";
    std::default_random_engine eng(seed);

    LOOP_NETWORK
    {
        for (auto& conn : nn.connections)
        {
            conn.second = uniform(eng);
        }
    }
}

// Clear all input data from neurons, but leave learned configuration intact
void Network::resetNeuronInputs()
{
    LOOP_NETWORK
    {
        nn.resetInput();
    }
}

// Set the activation function for a layer
void Network::setLayerActivationFunc(int layerIdx, activationFunc_t func)
{
    layer_t& layer = layers.at(layerIdx);

    for (Neuron& nn : layer)
    {
        nn.outputFunc = func;
    }
}

// Train network on the provided data
void Network::train(const dataset_t& input, const dataset_t& correct, double learningRate)
{
    process(input);

    // Compute deltas for all layers
    assignDeltas(layers.back(), correct);
    for (int i = layers.size() - 2; i >= 0; i--)
    {
        assignDeltas(layers[i]);
    }

    // Apply changes
    for (layer_t& layer : layers)
    {
        for (Neuron& nn : layer)
        {
            for (auto& conn : nn.connections)
            {
                double deriv = conn.first->data.delta * nn.data.output;

                // and it all comes down to this
                conn.second -= learningRate * deriv;
            }
        }
    }
}

// Train network on a batch of provided data
void Network::train(const std::vector<dataset_t>& input, const std::vector<dataset_t>& correct, double learningRate)
{
    // Allocate space for changes to be made
    //  layers    neurons    weights
    std::vector<std::vector<dataset_t>> changes(layers.size());
    for (unsigned i = 0; i < changes.size(); i++)
    {
        changes[i] = std::vector<dataset_t>(layers[i].size());
        for (unsigned j = 0; j < changes[i].size(); j++)
        {
            changes[i][j] = dataset_t(layers[i][j].connections.size());
        }
    }
    
    // Loop through all training sets
    for (unsigned setIdx = 0; setIdx < input.size(); setIdx++)
    {
        process(input[setIdx]);

        assignDeltas(layers.back(), correct[setIdx]);
        for (int i = layers.size() - 2; i >= 0; i--)
        {
            assignDeltas(layers[i]);
        }

        // Add desired changes to 'changes' vector
        for (unsigned i = 0; i < changes.size(); i++)
        {
            for (unsigned j = 0; j < changes[i].size(); j++)
            {
                const Neuron& nn = layers[i][j];
                for (unsigned k = 0; k < changes[i][j].size(); k++)
                {
                    const auto& conn = nn.connections[k];

                    changes[i][j][k] += conn.first->data.delta * nn.data.output;
                }
            }
        }
    }

    // Apply changes
    double batchSize = input.size();
    for (unsigned i = 0; i < changes.size(); i++)
    {
        for (unsigned j = 0; j < changes[i].size(); j++)
        {
            for (unsigned k = 0; k < changes[i][j].size(); k++)
            {
                layers[i][j].connections[k].second -= learningRate * changes[i][j][k] / batchSize;
            }
        }
    }
}

// Get output from the provided data, modifying states (but not configurations)
// of all neurons in the process
dataset_t Network::process(const dataset_t& input)
{
    exit_if_bad_vecs(layers, layers, "no layers");
    exit_if_bad_vecs(input, layers.front(), "input size doesn't match input layer size");

    // Check if layers exist
    if (!layers.size())
    {
        throw std::runtime_error("No layers exist -- can't process with an empty network!");
    }

    // Wipe data for the current run
    resetNeuronInputs();

    auto& layerInput = layers.front();

    // Provide input to first layer
    for (unsigned i = 0; i < input.size(); i++)
    {
        layerInput[i].input(input[i]);
    }

    // Perform calculations
    LOOP_NETWORK
    {
        nn.fire();
    }
    
    // Get output
    auto& layerOutput = layers.back();
    dataset_t result(layerOutput.size());
    for (unsigned i = 0; i < layerOutput.size(); i++)
    {
        result[i] = layerOutput[i].computeOutput(false);

        //std::cout << " -> nn #" << i << " = " << result[i] << "\n";
    }

    return result;
}

void Network::assignDeltas(layer_t& layer)
{
    for (Neuron& nn : layer)
    {
        double sum = 0;

        for (const auto& conn : nn.connections)
        {
            sum += conn.second * conn.first->data.delta;
        }

        nn.data.delta = nn.data.d_output * sum;
    }
}

void Network::assignDeltas(layer_t& layerLast, const dataset_t& correct)
{
    exit_if_bad_vecs(layerLast, correct, "training data");

    for (unsigned i = 0; i < layerLast.size(); i++)
    {
        Neuron& nn = layerLast[i];

        nn.data.delta = (nn.data.output - correct[i]) * nn.data.d_output;
    }
}

void Network::printAll() const
{
    for (unsigned i = 0; i < layers.size(); i++)
    {
        std::cout << "\033[1;31mLayer #" << i << "\033[0m\n";
        for (unsigned j = 0; j < layers[i].size(); j++)
        {
            const Neuron& nn = layers[i][j];
            std::cout << " -> Neuron [" << j << "]:\n"
                      << "   -> weights: " << nn.connections << "\n"
                      << "   -> delta: " << nn.data.delta << "\n"
                      << "   -> activation: " << nn.data.activation << "\n"
                      << "   -> output: " << nn.data.output << "\n";
        }
    }
    std::cout << "\n";
}

void Network::addLayer(int neurons, double defaultBiasFactor)
{
    // Check neuron count
    if (neurons < 1)
    {
        throw std::runtime_error("Number of neurons in a layer must be positive!");
    }

    layers.push_back(layer_t(neurons));
    layer_t& layerNew = layers.back();

    for (int i = 0; i < neurons; i++)
    {
        Neuron& nn_new = layerNew[i];
        nn_new.layerID = i;
        nn_new.outputFunc = identity;

        if (layers.size() > 1)
        {
            // Set necessary fields for newly added neuron
            layer_t& layerPrev = layers[layers.size() - 2];
            nn_new.layerPrev = &layerPrev;

            // Add connections to all neurons of the previous layer
            for (Neuron& nn_prev : layerPrev)
            {
                nn_prev.connections.push_back({&nn_new, 0});
                nn_prev.bias = defaultBiasFactor * layerPrev.size();
                nn_prev.outputFunc = rectifiedLeaky;
            }
        }
    }
}

//         //
// HELPERS //
//         //

double sigmoid(const double& v, const bool& deriv)
{
    if (deriv)
        return sigmoid(v, false) * (1.0 - sigmoid(v, false));
    else
        return 1.0 / (1.0 + std::exp(-v));
}

double identity(const double& v, const bool& deriv)
{
    if (deriv)
        return 1;
    else
        return v;
}

double rectified(const double& v, const bool& deriv)
{
    if (deriv)
        return (v < 0) ? 0 : 1;
    else
        return std::max(v, 0.0);
}

double rectifiedLeaky(const double& v, const bool& deriv)
{
    constexpr static const double leak = 0.01;
    if (deriv)
        return (v < 0) ? leak : 1;
    else
        return (v < 0) ? v * leak : v;
}

double computeMSE(const dataset_t& correct, const dataset_t& actual)
{
    // Check for sizes
    exit_if_bad_vecs(correct, actual, "correct and actual vectors must be the same size");

    double sum = 0;

    for (unsigned i = 0; i < correct.size(); i++)
    {
        sum += std::pow(correct[i] - actual[i], 2);
    }

    return sum / correct.size();
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::pair<Neuron*, double>>& weights)
{
    os << "{ ";
    for (const auto& conn : weights)
    {
        os << conn.second << " ";
    }
    return os << "}";
}

} // namespace znet