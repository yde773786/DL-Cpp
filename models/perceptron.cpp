#include "model.hpp"
#include <fstream>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

Perceptron::Perceptron(Node* activation, Node* loss, int input_size) : Model(loss){
    for (int i = 0; i < input_size; i++){
        input.push_back(new ChildlessNode(0));
    }

    graph = new ComputationalGraph();

    output = {new ChildlessNode(0)};
    target = {new ChildlessNode(0)};

    graph->add_connection(loss, output[0]);
    graph->add_connection(loss, target[0]);

    s1 = new FCSegment(input, output, activation, graph);
}

void Perceptron::forward(){
    graph->forward();
}

void Perceptron::backward(vector<double> target_val){

    for(int i = 0; i < target_val.size(); i++){
        target[i]->value = target_val[i];
    }

    graph->backward();
}

void Perceptron::load_weights(string weights_path){
    
    ifstream weights_file(weights_path);

    LOG_DEBUG("Input size: %ld", input.size());

    // Load bias and weights
    for(int i = 0; i < input.size(); i++){
        weights_file.read((char*)&s1->weights[i][0]->value, sizeof(double));
        LOG_DEBUG("Weight: %f", s1->weights[i][0]->value);
    }

    weights_file.read((char*)&s1->bias[0]->value, sizeof(double));
    LOG_DEBUG("Bias: %f", s1->bias[0]->value);
}