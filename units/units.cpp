#include "units.hpp"
#include <random>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

using namespace std;

FCSegment::FCSegment(vector<Node*> &n1, vector<Node*> &n2, Node* activation, ComputationalGraph &graph) : n1(n1), n2(n2), activation(activation){
    this->bias = vector<ChildlessNode*>(n2.size());
    this->weights = vector<vector<ChildlessNode*>>(n1.size(), vector<ChildlessNode*>(n2.size()));

    // We assume that n1 is already connected to the computational graph

    LOG_DEBUG("n1 size: %ld", n1.size());
    LOG_DEBUG("n2 size: %ld", n2.size());

    // Use He initialization for weights
    for(int i = 0; i < n1.size(); i++){
        for(int j = 0; j < n2.size(); j++){
            double r = (rand() % 1000) / 1000.0;
            
            this->weights[i][j] = new ChildlessNode(sqrt(2.0 / n1.size()) * r);
            graph.add_node(this->weights[i][j]);

            LOG_DEBUG("Weight: %f", this->weights[i][j]->value);
        }
    }

    // Use He initialization for bias
    for(int i = 0; i < n2.size(); i++){
        double r = (rand() % 1000) / 1000.0;

        this->bias[i] = new ChildlessNode(sqrt(2.0 / n1.size()) * r);
        graph.add_node(this->bias[i]);

        LOG_DEBUG("Bias: %f", this->bias[i]->value);
    }

    // Set up the computational graph of the fully connected segment
    for(int i = 0; i < n1.size(); i++){

        AddNode* add = new AddNode(0);
        graph.add_node(add);

        for(int j = 0; j < n2.size(); j++){
            // w_ij * n1_i
            MulNode* mul = new MulNode(1);
            graph.add_node(mul);
            graph.add_connection(mul, n1[i]);
            graph.add_connection(mul, this->weights[i][j]);

            graph.add_connection(add, mul);
        }

        // + b_i
        graph.add_connection(add, this->bias[i]);

        // Apply activation function
        graph.add_node(activation);
        graph.add_connection(activation, add);

        // Populate n2 with the final layer (activation nodes)
        n2[i] = activation;
    }
}