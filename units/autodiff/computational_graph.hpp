#include "node.hpp"
#include <unordered_set>
#include <stack>

using namespace std;

enum direction
{
    FORWARD = 0,
    BACKWARD = 1
};

class ComputationalGraph {
    public:
        unordered_set<Node*> nodes;

        void add_node(Node* node);
        void add_connection(Node* parent, Node* child);

        void forward();
        void backward();
        void reset_grad();

    private:
        // topological sort of the graph. pop from the 
        stack<Node*> topological_sort(direction dir);
        void topological_sort_helper(Node* node, unordered_set<Node*>& visited, stack<Node*>& sortedNodes, direction dir);

};