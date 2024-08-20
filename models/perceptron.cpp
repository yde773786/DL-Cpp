#include "model.hpp"
#include <fstream>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

Perceptron::Perceptron(Node* activation, Node* loss, int input_size, double learning_rate) : Model(loss){

    graph = new ComputationalGraph();
    this->learning_rate = learning_rate;

    for (int i = 0; i < input_size; i++){
        input.push_back(new ChildlessNode(0));
        graph->add_node(input[i]);
    }

    target = {new ChildlessNode(0)};

    // Output layer is populated in FCSegment. We need to create a dummy node here
    output = {NULL};

    graph->add_node(target[0]);
    graph->add_node(loss);

    s1 = new FCSegment(input, output, activation, graph);

    graph->add_connection(loss, output[0]);
    graph->add_connection(loss, target[0]);

    ((LossNode*)loss)->add_output_target_pair(output[0], target[0]);
}

void Perceptron::forward(){
    graph->forward();
}

void Perceptron::backward(){
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