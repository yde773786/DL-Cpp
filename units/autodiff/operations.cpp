#include "operations.hpp"
#include <cmath>

void AddNode::forward() {
    // AddNode is defined s.t it has two children

    for(int i = 0; i < this->num_elements; i++) {
        this->value[i] = 0;
        for(auto it = this->children.begin(); it != this->children.end(); it++) {
            this->value[i] += (*it)->value[i];
        }
    }
}

void AddNode::backward(Node* child) {

    for(int i = 0; i < this->num_elements; i++) {
        child->gradient[i] += this->gradient[i];
    }
};

void MulNode::forward() {
    // MulNode is defined s.t it has two children

    for(int i = 0; i < this->num_elements; i++) {
        this->value[i] = 1;
        for(auto it = this->children.begin(); it != this->children.end(); it++) {
            this->value[i] *= (*it)->value[i];
        }
    }
}

void MulNode::backward(Node* child) {

    for(int i = 0; i < this->num_elements; i++) {
        if (child->value[i] != 0) {
            child->gradient[i] += (this->value[i] / child->value[i]) * this->gradient[i];
        }
    }
};

void MatMulNode::forward() {
    // MatMulNode is defined s.t it has two children

    // Child 1: (m x n), Child 2: (n x p) => Output: (m x p)

    auto it = this->children.begin();
    Node *child1 = *it;
    it++;
    Node *child2 = *it;

    int m = child1->shape[0], n = child1->shape[1], p = child2->shape[1];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            this->value[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                this->value[i * p + j] += child1->value[i * n + k] * child2->value[k * p + j];
            }
        }
    }
}

void MatMulNode::backward(Node* child) {
    // MatMulNode is defined s.t it has two children

    // Child 1: (m x n), Child 2: (n x p) => Output: (m x p)

    auto it = this->children.begin();
    Node *child1 = *it;
    it++;
    Node *child2 = *it;

    int m = child1->shape[0], n = child1->shape[1], p = child2->shape[1];
    if (child == child1) {
        // dL/dC_1 = dL/dO * C_2^T
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    child1->gradient[i * n + j] += this->gradient[i * p + k] * child2->value[j * p + k];
                }
            }
        }
    } else if (child == child2) {
        // dL/dC_2 = C_1^T * dL/dO
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < m; k++) {
                    child2->gradient[i * p + j] += child1->value[k * n + i] * this->gradient[k * p + j];
                }
            }
        }
    }
}

void ChildlessNode::forward() {
    // do nothing
}

void ChildlessNode::backward(Node* child) {
    // do nothing
};