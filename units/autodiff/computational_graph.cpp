#include "computational_graph.hpp"
#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

void ComputationalGraph::add_node(Node* node) {
    nodes.insert(node);
}

void ComputationalGraph::add_connection(Node* parent, Node* child) {
    parent->add_child(child);
    child->add_parent(parent);
}

stack<Node*> ComputationalGraph::topological_sort(direction dir) {
    stack<Node*> sortedNodes;
    unordered_set<Node*> visited;

    for(auto it = nodes.begin(); it != nodes.end(); it++) {
        if(visited.count(*it) == 0){
            topological_sort_helper(*it, visited, sortedNodes, dir);
        }
    }

    return sortedNodes;
}

void ComputationalGraph::topological_sort_helper(Node* node, unordered_set<Node*>& visited, stack<Node*>& sortedNodes, direction dir) {
    visited.insert(node);

    if(dir == BACKWARD) {
        for(auto it = node->children.begin(); it != node->children.end(); it++) {
            if(visited.count(*it) == 0) {
                topological_sort_helper(*it, visited, sortedNodes, dir);
            }
        }
    } else {
        for(auto it = node->parents.begin(); it != node->parents.end(); it++) {
            if(visited.count(*it) == 0) {
                topological_sort_helper(*it, visited, sortedNodes, dir);
            }
        }
    }

    sortedNodes.push(node);
}

void ComputationalGraph::forward() {
    stack<Node*> sortedNodes = topological_sort(FORWARD);

    while(!sortedNodes.empty()) {
        Node* node = sortedNodes.top();
        node->forward();
        sortedNodes.pop();
    }
}

void ComputationalGraph::backward() {
    stack<Node*> sortedNodes = topological_sort(BACKWARD);

    while(!sortedNodes.empty()) {
        Node* node = sortedNodes.top();
        // find the gradient of all the children w.r.t. this node
        for(auto it = node->children.begin(); it != node->children.end(); it++) {
            node->backward(*it);
        }
        sortedNodes.pop();
    }
}

void ComputationalGraph::save_grad() {
    for(auto it = nodes.begin(); it != nodes.end(); it++) {
        (*it)->apply_grad += (*it)->gradient;
        (*it)->gradient = 0;
    }
}

void ComputationalGraph::apply_grad(double batch_size, double learning_rate = 0.01) {
    for(auto it = nodes.begin(); it != nodes.end(); it++) {
        (*it)->value -= learning_rate * ((*it)->apply_grad / batch_size);
        (*it)->apply_grad = 0;
    }
}