#include "units.hpp"
#include <cmath>

using namespace std;

Node::Node(double value) {
    this->value = value;
    this->gradient = 0;
    this->parents = set<Node*>();
    this->children = set<Node*>();
}

void Node::add_parent(Node* parent) {
    this->parents.insert(parent);
}

void Node::add_child(Node* child) {
    this->children.insert(child);
}

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

void AddNode::forward() {
    for(auto it = this->children.begin(); it != this->children.end(); it++) {
        this->value += (*it)->value;
    }
}

void AddNode::backward(Node* child) {
    child->gradient += 1;
};

void MulNode::forward() {
    this->value = 1;
    for(auto it = this->children.begin(); it != this->children.end(); it++) {
        this->value *= (*it)->value;
    }
}

void MulNode::backward(Node* child) {
    child->gradient += value / child->value;
};