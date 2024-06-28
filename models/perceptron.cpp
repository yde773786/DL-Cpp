#include "model.hpp"

Perceptron::Perceptron(activation_function activation, loss_function loss, int input_size) : Model(activation, loss){
    for (int i = 0; i < input_size; i++){
        neuron new_neuron;
        new_neuron.activation = 0;
        input.push_back(new_neuron);
    }
}

void Perceptron::forward(){

    vector<neuron> middle_neuron = {middle};

    Segment s1(input, middle_neuron, activation);
    Segment s2(middle_neuron, output, activation);

    s1.forward();
    s2.forward();
}