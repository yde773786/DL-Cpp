#include <vector>
#include <unordered_map>
#include <autodiff/node.hpp>
#include <autodiff/activations.hpp>
#include <autodiff/operations.hpp>
#include <autodiff/computational_graph.hpp>
#include <autodiff/loss_fns.hpp>
#include <string>

using namespace std;

// Represents a Fully Connected Segment between two layers of neurons
class FCSegment{

    public:

        vector<Node*> &n1; // First layer of neurons
        vector<Node*> &n2; // Second layer of neurons. This layer will be re-populated by FCSegment.

        vector<ChildlessNode*> bias; // Bias for each neuron in the second layer
        vector<vector<ChildlessNode*>> weights; // Weights between the two layers

        Node* activation;
        Node* loss;

        FCSegment(vector<Node*> &n1, vector<Node*> &n2, Node* activation, ComputationalGraph* graph);
};