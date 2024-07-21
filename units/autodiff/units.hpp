#include <unordered_set>

using namespace std;

struct PropogationFunctionHandler {
    double (*forward)(unordered_set<Node*>);
    double (*backward)(Node*, unordered_set<Node*>);
};

// Activation function handlers
PropogationFunctionHandler sigmoid_handler;
PropogationFunctionHandler ReLU_handler;
PropogationFunctionHandler tanh_handler;

// Operation function handlers
PropogationFunctionHandler add_handler;
PropogationFunctionHandler multiply_handler;

// Loss function handlers
PropogationFunctionHandler squared_error_handler;

class Node {
    public:
        double value;
        double gradient;
        unordered_set<Node*> parents;
        unordered_set<Node*> children;
        PropogationFunctionHandler propogation;

        Node(double);
        void add_parent(Node);
        void add_child(Node);

        void forward();

        // The backward function depends on which child is calling it
        void backward(Node* child);
};