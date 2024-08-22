#include <set>
#pragma once

using namespace std;

class Node {
    public:
        double value;
        double gradient;
        set<Node*> parents;
        set<Node*> children;
        double apply_grad = 0;

        Node(double value) : value(value), gradient(0), parents(set<Node*>()), children(set<Node*>()) {}

        void add_parent(Node* parent) {
            this->parents.insert(parent);
        }
        
        void add_child(Node* child) {
            this->children.insert(child);
        }

        virtual void forward() = 0;

        // partial derivative of the node with respect to the child. Assign the gradient to the child.
        virtual void backward(Node* child) = 0;
};