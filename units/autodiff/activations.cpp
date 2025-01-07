#include "activations.hpp"
#include <cmath>

void SigmoidNode::forward() {
    // SigmoidNode is defined s.t it has one child

    auto child = *this->children.begin();
    for(int i = 0; i < this->num_elements; i++) {
        this->value[i] = 1 / (1 + exp(-child->value[i]));
    }
}

void SigmoidNode::backward(Node* child) {
    for(int i = 0; i < this->num_elements; i++) {
        child->gradient[i] += this->value[i] * (1 - this->value[i]) * this->gradient[i];
    }
};

void TanhNode::forward() {
    // TanhNode is defined s.t it has one child

    auto child = *this->children.begin();
    for(int i = 0; i < this->num_elements; i++) {
        this->value[i] = tanh(child->value[i]);
    }
}

void TanhNode::backward(Node* child) {
    for(int i = 0; i < this->num_elements; i++) {
        child->gradient[i] += (1 - pow(this->value[i], 2)) * this->gradient[i];
    }
};

void ReLUNode::forward() {
    // ReLUNode is defined s.t it has one child

    auto child = *this->children.begin();
    for(int i = 0; i < this->num_elements; i++) {
        this->value[i] = fmax(0, child->value[i]);
    }
}

void ReLUNode::backward(Node* child) {
    for(int i = 0; i < this->num_elements; i++) {
        child->gradient[i] += (child->value[i] > 0) * this->gradient[i];
    }
};