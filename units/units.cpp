#include "units.hpp"
#include <random>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

using namespace std;

Segment::Segment(vector<neuron> &n1, vector<neuron> &n2, activation_function activation): n1(n1), n2(n2){

    this->bias = vector<weight>(n2.size());
    this->weights = vector<vector<weight>>(n1.size(), vector<weight>(n2.size()));

    LOG_DEBUG("n1 size: %ld", n1.size());
    LOG_DEBUG("n2 size: %ld", n2.size());

    // Use He initialization for weights
    for(int i = 0; i < n1.size(); i++){
        for(int j = 0; j < n2.size(); j++){

            double r = (rand() % 1000) / 1000.0;
            this->weights[i][j].value = sqrt(2.0 / n1.size()) * r;
            LOG_DEBUG("Weight: %f", this->weights[i][j].value);
        }
    }

    this->activation = activation;
}

// Forward pass N_2 = act(W^T * N_1)
void Segment::forward(){
    for(int j = 0; j < n2.size(); j++){
        double sum = 0;
        for(int i = 0; i < n1.size(); i++){
            LOG_DEBUG("Neuron Activation: %f", n1[i].activation);
            LOG_DEBUG("Weight: %f", weights[i][j].value);
            sum += n1[i].activation * weights[i][j].value;
        }

        sum += bias[j].value;
        
        LOG_DEBUG("Sum: %f", sum);
        n2[j].activation = activation(sum);
    }
}