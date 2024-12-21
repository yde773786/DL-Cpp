#include "model.hpp"
#include <fstream>
#include <cassert>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

MLP::MLP(string loss_str, vector<MLPUnit> mlp_unit, double learning_rate){
    
    graph = new ComputationalGraph();
    this->learning_rate = learning_rate;

    if(mlp_unit.size() < 2){
        cerr << "MLP should have at least 2 units. Create a Perceptron if there are no hidden layers" << endl;
        return;
    }

    for (int i = 0; i < mlp_unit[0].input_size; i++){
        input.push_back(new ChildlessNode(0));
        input[i]->set_id("Input " + to_string(i));
        graph->add_node(input[i]);
    }

    vector<Node*> hidden_layer(mlp_unit[0].output_size, NULL);
    segments.push_back(new FCSegment(input, hidden_layer, mlp_unit[0].activation, graph));
    hidden_layers.push_back(hidden_layer);

    for(int i = 1; i < mlp_unit.size(); i++){
        vector<Node*> hidden_layer(mlp_unit[i].output_size, NULL);
        segments.push_back(new FCSegment(hidden_layers[i-1], hidden_layer, mlp_unit[i].activation, graph));
        hidden_layers.push_back(hidden_layer);
    }

    output = hidden_layers.back();
    hidden_layers.pop_back();

    LossNode * loss = LOSS_FUNCTIONS[loss_str]();
    this->loss = loss;

    for (int i = 0; i < mlp_unit.back().output_size; i++){
        target.push_back(new ChildlessNode(0));
        graph->add_node(target[i]);
        graph->add_connection(loss, output[i]);
        graph->add_connection(loss, target[i]);
        loss->add_output_target_pair(output[i], target[i]);
    }
}

void MLP::forward(){
    graph->forward();
}

void MLP::backward(){
    graph->backward();
}

void MLP::load_weights(string weights_path){
    
    ifstream weights_file(weights_path);

    for(int i = 0; i < segments.size(); i++){
        for(int j = 0; j < segments[i]->weights.size(); j++){
            for(int k = 0; k < segments[i]->weights[j].size(); k++){
                weights_file.read((char*)&segments[i]->weights[j][k]->value, sizeof(double));
                LOG_DEBUG("Weight: Layer %i, i %i, j %i: %f", i + 1, j + 1, k + 1 ,segments[i]->weights[j][k]->value);
            }
        }

        for(int j = 0; j < segments[i]->bias.size(); j++){
          weights_file.read((char*)&segments[i]->bias[j]->value, sizeof(double));
          LOG_DEBUG("Bias: Layer %i, i %i: %f", i + 1, j + 1, segments[i]->bias[j]->value);
        }
    }

}