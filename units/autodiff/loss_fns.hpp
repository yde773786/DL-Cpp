#include "node.hpp"

class MSENode : public Node {
    public:
        MSENode(double value) : Node(value) {}

        void forward();
        void backward(Node* child);
};