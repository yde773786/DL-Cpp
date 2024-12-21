#include "units.hpp"
#include <random>
#include <time.h>

#include "../globals.hpp"

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

using namespace std;

std::unordered_map<std::string, std::function<Node*()>> ACTIVATION_FUNCTIONS = {
    {"sigmoid", []() { return new SigmoidNode(0); }},
    {"tanh", []() { return new TanhNode(0); }},
    {"relu", []() { return new ReLUNode(0); }},
};

std::unordered_map<std::string, std::function<LossNode*()>> LOSS_FUNCTIONS = {
    {"mse", []() { return new MSENode(0); }},
};

FCSegment::FCSegment(vector<Node*> &n1, vector<Node*> &n2, string activation_str, ComputationalGraph* graph) : n1(n1), n2(n2){
    this->bias = vector<ChildlessNode*>(n2.size());
    this->weights = vector<vector<ChildlessNode*>>(n2.size(), vector<ChildlessNode*>(n1.size()));

    // We assume that n1 is already connected to the computational graph

    LOG_DEBUG("n1 size: %ld", n1.size());
    LOG_DEBUG("n2 size: %ld", n2.size());

    /* initialize random seed: */
    srand ( time(NULL) );

    ++GLOBAL_INCREMENT;

    // Use He initialization for weights
    for(int i = 0; i < n2.size(); i++){
        for(int j = 0; j < n1.size(); j++){
            double r = (rand() % 1000) / 1000.0;
            
            this->weights[i][j] = new ChildlessNode(sqrt(2.0 / n1.size()) * r);
            this->weights[i][j]->set_id("Layer " + to_string(GLOBAL_INCREMENT) + " Weight " + to_string(i + 1) + " " + to_string(j + 1));
            graph->add_node(this->weights[i][j]);

            LOG_DEBUG("Weight: %f", this->weights[i][j]->value);
        }
    }

    // Use He initialization for bias
    for(int i = 0; i < n2.size(); i++){
        double r = (rand() % 1000) / 1000.0;

        this->bias[i] = new ChildlessNode(sqrt(2.0 / n1.size()) * r);
        this->bias[i]->set_id("Bias " + to_string(GLOBAL_INCREMENT) + " " + to_string(i + 1));
        graph->add_node(this->bias[i]);

        LOG_DEBUG("Bias: %f", this->bias[i]->value);
    }

    // Set up the computational graph of the fully connected segment
    for(int i = 0; i < n2.size(); i++){

        AddNode* add = new AddNode(0);
        add->set_id("Layer " + to_string(GLOBAL_INCREMENT + 1) + " Node " + to_string(i + 1));
        graph->add_node(add);

        for(int j = 0; j < n1.size(); j++){
            // w_ij * n1_j
            MulNode* mul = new MulNode(1);
            mul->set_id("Layer " + to_string(GLOBAL_INCREMENT) + " W*H " + to_string(i + 1) + " " + to_string(j + 1));
            graph->add_node(mul);
            graph->add_connection(mul, n1[j]);
            graph->add_connection(mul, this->weights[i][j]);

            graph->add_connection(add, mul);
        }

        // + b_i
        graph->add_connection(add, this->bias[i]);

        Node* activation = ACTIVATION_FUNCTIONS[activation_str]();
        activation->set_id("Layer " + to_string(GLOBAL_INCREMENT) + " Activation " + to_string(i + 1));

        graph->add_node(activation);
        graph->add_connection(activation, add);

        // Populate n2 with the final layer (activation nodes)
        n2[i] = activation;
    }
}