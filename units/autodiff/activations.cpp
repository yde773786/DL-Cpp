#include "activations.hpp"
#include <cmath>

void SigmoidNode::forward() {
    auto child = *this->children.begin();
    this->value = 1 / (1 + exp(-child->value));
}

void SigmoidNode::backward(Node* child) {
    child->gradient += value * (1 - value);
};

void TanhNode::forward() {
    auto child = *this->children.begin();
    this->value = tanh(child->value);
}

void TanhNode::backward(Node* child) {
    child->gradient += 1 - pow(value, 2);
};

void ReLUNode::forward() {
    auto child = *this->children.begin();
    this->value = max(0.0, child->value);
}

void ReLUNode::backward(Node* child) {
    child->gradient += value > 0 ? 1 : 0;
};