#include "units.hpp"
#include <random>

using namespace std;

Segment::Segment(vector<neuron> &n1, vector<neuron> &n2, activation_function activation){
    this->n1 = n1;
    this->n2 = n2;

    this->bias = vector<weight>(n2.size());
    this->weights = vector<vector<weight>>(n1.size(), vector<weight>(n2.size()));

    // Use He initialization for weights
    for(int i = 0; i < n1.size(); i++){
        for(int j = 0; j < n2.size(); j++){

            double r = (rand() % 1000) / 1000.0;
            this->weights[i][j].value = sqrt(2.0 / n1.size()) * r;
        }
    }

    this->activation = activation;
}

// Forward pass N_2 = act(W^T * N_1)
void Segment::forward(){
    for(int j = 0; j < n2.size(); j++){
        double sum = 0;
        for(int i = 0; i < n1.size(); i++){
            sum += n1[i].activation * weights[i][j].value + bias[j].value;
        }
        n2[j].activation = activation(sum);
    }
}