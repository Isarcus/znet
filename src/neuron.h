#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstdint>

namespace znet
{
    struct Neuron;
    typedef std::vector<Neuron> layer_t;

    typedef double (*activationFunc_t)(const double&, const bool&);

    typedef struct activationData_t
    {
        double activation;
        double output;
        double d_output;

        double delta;
    } activationData_t;

    typedef struct Neuron
    {
        //         //
        // Members //
        //         //
        uint64_t layerID;
        layer_t* layerPrev;
        activationFunc_t outputFunc;

        std::vector<std::pair<Neuron*, double>> connections;
        double bias;

        activationData_t data;
        
        //           //
        // Functions //
        //           //

        Neuron();

        void input(double val);
        void resetInput();
        void fire();

        double computeOutput(bool deriv) const;
    } Neuron;
}

#endif