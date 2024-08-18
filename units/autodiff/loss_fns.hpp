#include "node.hpp"
#include <unordered_map>
#include <tuple>

class LossNode : public Node {
    public:
        // Maps the output/target node to the pair of the other node and the sign (y - y' or y' - y)
        unordered_map<Node*, pair<Node*, int>> output_target_pair;

        LossNode(double value) : Node(value) {}

        virtual void forward() = 0;
        virtual void backward(Node* child) = 0;
        virtual void add_output_target_pair(Node* output, Node* target) = 0;
};

class MSENode : public LossNode {
    public:

        MSENode(double value) : LossNode(value) {}

        void forward() override;
        void backward(Node* child) override;
        void add_output_target_pair(Node* output, Node* target) override;
};