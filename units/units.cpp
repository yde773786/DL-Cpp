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

FCSegment::FCSegment(Node* n1, Node* n2, ComputationalGraph* graph){

    // n1 is B x n1_dim
    // n2 is B x n2_dim
    // bias is n2_dim
    // weights is n2_dim x n1_dim

    int n1_dim = n1->shape[1];
    int n2_dim = n2->shape[1];
    int batch_size = n1->shape[0];

    this->bias = ChildlessNode({n2_dim});
    this->weights = ChildlessNode({n2_dim, n1_dim});

    graph->add_node(this->bias);
    graph->add_node(this->weights);

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

            this->weights[i * n1.size() + j] = sqrt(2.0 / n1.size()) * r;
            LOG_DEBUG("Weight: %f", this->weights[i * n1.size() + j]);
        }
    }

    // Use He initialization for bias
    for(int i = 0; i < n2.size(); i++){
        double r = (rand() % 1000) / 1000.0;

        this->bias[i] = sqrt(2.0 / n1.size()) * r;
        LOG_DEBUG("Bias: %f", this->bias[i]);
    }

    // Set up the computational graph of the fully connected segment

    MatMulNode* matmul = new MatMulNode({batch_size, n2_dim});
    graph->add_node(matmul);
    graph->add_connection(matmul, n1);
    graph->add_connection(matmul, this->weights);

    AddNode* add = new AddNode({batch_size, n2_dim});
    graph->add_node(add);
    graph->add_connection(add, matmul);
    graph->add_connection(add, this->bias);

    graph->add_node(n2);
    graph->add_connection(n2, add);

}