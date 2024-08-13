#include "node.hpp"

class AddNode : public Node {
    public:
        AddNode(double value) : Node(value) {}

        void forward();
        void backward(Node* child);
};

class MulNode : public Node {
    public:
        MulNode(double value) : Node(value) {}

        void forward();
        void backward(Node* child);
};

class ChildlessNode : public Node {
    public:
        ChildlessNode(double value) : Node(value) {}

        void forward();
        void backward(Node* child);
};