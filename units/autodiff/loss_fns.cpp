#include "loss_fns.hpp"
#include <cmath>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

// Children contain the output nodes and target nodes in order.
// First half are output nodes, second half are target nodes.
// [y'1, y'2, y'3, y'4... y'N, y1, y2, y3, y4... yN]
// The loss is calculated as ((y'1 - y1)^2 + (y'2 - y2)^2 + ... + (y'N - yN)^2) / N
void MSENode::forward(){
    auto it = this->children.begin();
    Node* output = *it;
    it++;
    Node* target = *it;

    int vec_len = output->num_elements;

    double sum = 0;
    for(int i = 0; i < vec_len; i++){
        sum += pow(output->value[i] - target->value[i], 2);
    }

    this->value[0] = sum / vec_len;
}

// If y'x : 2 * (y'x - yx) / N
// If yx : 2 * (yx - y'x) / N
// the gradient is calculated as 2 * (y'x - yx) / N (for output nodes)
// grad wrt target not considered
void MSENode::backward(Node* child){

    auto it = this->children.begin();
    Node* output = *it;
    it++;
    Node* target = *it;

    int vec_len = output->num_elements;

    for(int i = 0; i < vec_len; i++){
        output->gradient[i] += 2 * (output->value[i] - target->value[i]) / vec_len;
    }
}
