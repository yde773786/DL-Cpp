#include "operations.hpp"
#include <cmath>

void AddNode::forward() {
    this->value = 0;
    for(auto it = this->children.begin(); it != this->children.end(); it++) {
        this->value += (*it)->value;
    }
}

void AddNode::backward(Node* child) {
    child->gradient += 1 * this->gradient;
};

void MulNode::forward() {
    this->value = 1;
    for(auto it = this->children.begin(); it != this->children.end(); it++) {
        this->value *= (*it)->value;
    }
}

void MulNode::backward(Node* child) {
    child->gradient += (value / child->value) * this->gradient;
};

void ChildlessNode::forward() {
    // do nothing
}

void ChildlessNode::backward(Node* child) {
    // do nothing
};