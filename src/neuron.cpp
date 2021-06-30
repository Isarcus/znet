#include "neuron.h"

namespace znet
{

Neuron::Neuron()
    : layerPrev(nullptr)
    , outputFunc(nullptr)
    , bias(0)
    , data{0, 0, 0, 0}
{}

void Neuron::input(double val)
{
    data.activation += val;
}

void Neuron::resetInput()
{
    data.activation = 0;
    data.delta = 0;
    data.d_output = 0;
    data.output = 0;
}

void Neuron::fire()
{
    data.output = computeOutput(false);
    data.d_output = computeOutput(true);
    for (const auto& conn : connections)
    {
        // Send weighted activations
        conn.first->input(data.output * conn.second);
    }
}

double Neuron::computeOutput(bool deriv) const
{
    return outputFunc(data.activation, deriv);
}

} // namespace znet