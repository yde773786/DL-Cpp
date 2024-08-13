#include "node.hpp"

class SigmoidNode : public Node {
    public:
        SigmoidNode(double value) : Node(value) {}

        void forward();
        void backward(Node* child);
};

class TanhNode : public Node {
    public:
        TanhNode(double value) : Node(value) {}

        void forward();
        void backward(Node* child);
};

class ReLUNode : public Node {
    public:
        ReLUNode(double value) : Node(value) {}

        void forward();
        void backward(Node* child);
};