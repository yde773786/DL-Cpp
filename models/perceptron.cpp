#include "../units/units.cpp"
#include "../units/activation.cpp"
#include "../units/loss.cpp"

class Perceptron
{
    public:
        
        vector<neuron> input;
        neuron middle;
        neuron output;
};