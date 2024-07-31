#include "units.hpp"
#include <unordered_set>

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

    private:
        set<Node*> topological_sort(direction dir);

};