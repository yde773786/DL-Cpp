#include "loss_fns.hpp"
#include <cmath>

// Children contain the output nodes and target nodes in order.
// First half are output nodes, second half are target nodes.
void MSENode::forward(){
    double sum = 0;
    set<Node*>::iterator start_child = children.begin();
    set<Node*>::reverse_iterator end_child = children.rbegin();

    while(std::distance(start_child, end_child.base()) > 0){
        sum += pow((*start_child)->value - (*end_child)->value, 2);
        start_child++;
        end_child++;
    }

    value = sum / children.size();
}

void MSENode::backward(Node* child){
    //TODO: Implement
}