#include "loss_fns.hpp"
#include <cmath>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

// Children contain the output nodes and target nodes in order.
// First half are output nodes, second half are target nodes.
// [y'1, y'2, y'3, y'4... y'N, y1, y2, y3, y4... yN]
void MSENode::forward(){
    double sum = 0;
    set<Node*>::iterator start_child = children.begin();
    set<Node*>::reverse_iterator end_child = children.rbegin();

    int vec_len = children.size() / 2;

    while(std::distance(start_child, end_child.base()) > 0){
        sum += pow((*start_child)->value - (*end_child)->value, 2);
        start_child++;
        end_child++;
    }

    value = sum / vec_len;
}

// If y'x : 2 * (y'x - yx) / N
// If yx : 2 * (yx - y'x) / N
void MSENode::backward(Node* child){
    int vec_len = children.size() / 2;

    int sign = output_target_pair[child].second;
    Node* output = output_target_pair[child].first;

    // sign will be 1 if the child is an output node, -1 if it is a target node
    child->gradient += 2 * ((child->value - output->value) * sign) / vec_len;
}

// y'x : yx
// yx : y'x
void MSENode::add_output_target_pair(Node* output, Node* target){
    output_target_pair[output] = make_pair(target, 1);
    output_target_pair[target] = make_pair(output, -1);
}