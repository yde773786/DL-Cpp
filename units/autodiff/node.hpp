#pragma once

#include <string>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

using namespace std;

class Node {

    private:
        double num_elements;

    public:
        double* value;
        double* gradient;

        vector<double> shape;

        set<Node*> parents;
        set<Node*> children;
        double apply_grad = 0;

        // id is used for debugging purposes
        string id = "";

        Node(double* value, vector<double> shape) {
            this->value = value;
            this->shape = shape;

            this->num_elements = 1;
            for (int i = 0; i < shape.size(); i++) {
                this->num_elements *= shape[i];
            }

            this->gradient = new double[ttl];
            for (int i = 0; i < ttl; i++) {
                this->gradient[i] = 0;
            }
        }

        Node(vector<double> shape) {
            this->shape = shape;

            // Allow (1, n) to be treated as (n)
            if (shape.size() == 2 && shape[0] == 1) {
                shape.erase(shape.begin());
            }


            this->num_elements = 1;
            for (int i = 0; i < shape.size(); i++) {
                this->num_elements *= shape[i];
            }

            this->value = new double[ttl];
            for (int i = 0; i < ttl; i++) {
                this->value[i] = 0;
            }

            this->gradient = new double[ttl];
            for (int i = 0; i < ttl; i++) {
                this->gradient[i] = 0;
            }
        }

        ~Node() {
            delete[] gradient;
        }

        void set_id(string id) {
            this->id = id;
        }

        void add_parent(Node* parent) {
            this->parents.insert(parent);
        }
        
        void add_child(Node* child) {
            this->children.insert(child);
        }

        virtual void forward() = 0;

        // partial derivative of the node with respect to the child. Assign the gradient to the child.
        virtual void backward(Node* child) = 0;
}