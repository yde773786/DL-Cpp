#include "model.hpp"
#include <fstream>

Perceptron::Perceptron(activation_function activation, loss_function loss, int input_size) : Model(activation, loss), s1(input, output, activation){
    for (int i = 0; i < input_size; i++){
        neuron new_neuron;
        new_neuron.activation = 0;
        input.push_back(new_neuron);
    }

    output = {{0}};
}

void Perceptron::forward(){
    s1.forward();
}

void Perceptron::load_weights(string weights_path){
    
    ifstream weights_file(weights_path);

    // Load bias and weights
    for(int i = 0; i < input.size(); i++){
        weights_file.read((char*)&s1.bias[0].value, sizeof(double));
        weights_file.read((char*)&s1.weights[i][0].value, sizeof(double));
    }
}