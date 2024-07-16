#include "model.hpp"
#include <fstream>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

Perceptron::Perceptron(activation_function activation, loss_function loss, int input_size) : Model(loss){
    for (int i = 0; i < input_size; i++){
        neuron new_neuron;
        new_neuron.activation = 0;
        input.push_back(new_neuron);
    }

    output = {{0}};
    s1 = new Segment(input, output, activation);
}

void Perceptron::forward(){
    s1->forward();
}

void Perceptron::load_weights(string weights_path){
    
    ifstream weights_file(weights_path);

    LOG_DEBUG("Input size: %ld", input.size());

    // Load bias and weights
    for(int i = 0; i < input.size(); i++){
        weights_file.read((char*)&s1->weights[i][0].value, sizeof(double));
        LOG_DEBUG("Weight: %f", s1->weights[i][0].value);
    }

    weights_file.read((char*)&s1->bias[0].value, sizeof(double));
    LOG_DEBUG("Bias: %f", s1->bias[0].value);
}