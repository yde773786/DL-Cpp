#include <vector>

using namespace std;

struct neuron
{
    double activation;
};

struct weight
{
    double value;
};

typedef double (*activation_function)(double);
typedef double (*loss_function)(vector<neuron>, vector<neuron>);

// Represents a connection between two layers of a neural network
class Segment{

    public:

        vector<neuron> n1; // First layer of neurons
        vector<neuron> n2; // Second layer of neurons

        vector<weight> bias; // Bias for each neuron in the second layer
        vector<vector<weight>> weights; // Weights between the two layers

        activation_function activation;

        Segment(vector<neuron> &n1, vector<neuron> &n2, activation_function activation);

        void forward();
};