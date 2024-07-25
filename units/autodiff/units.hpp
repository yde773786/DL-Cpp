#include <set>

using namespace std;

class Node {
    public:
        double value;
        double gradient;
        set<Node*> parents;
        set<Node*> children;

        Node(double);
        void add_parent(Node*);
        void add_child(Node*);

        virtual void forward() = 0;

        // partial derivative of the node with respect to the child. Assign the gradient to the child.
        virtual void backward(Node* child) = 0;
};

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