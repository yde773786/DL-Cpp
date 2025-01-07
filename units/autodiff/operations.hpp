#include "node.hpp"

class AddNode : public Node {
    public:
        AddNode(double* value, vector<double> shape) : Node(value, shape) {}
        AddNode(vector<double> shape) : Node(shape) {}

        void forward();
        void backward(Node* child);
};

class MulNode : public Node {
    public:
        MulNode(double* value, vector<double> shape) : Node(value, shape) {}
        MulNode(vector<double> shape) : Node(shape) {}

        void forward();
        void backward(Node* child);
};

class MatMulNode : public Node {
    public:
        MatMulNode(double* value, vector<double> shape) : Node(value, shape) {}
        MatMulNode(vector<double> shape) : Node(shape) {}

        void forward();
        void backward(Node* child);
};

class ChildlessNode : public Node {
    public:
        ChildlessNode(double* value, vector<double> shape) : Node(value, shape) {}
        ChildlessNode(vector<double> shape) : Node(shape) {}

        void forward();
        void backward(Node* child);
};