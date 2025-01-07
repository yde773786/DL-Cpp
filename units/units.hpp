#include <vector>
#include <unordered_map>
#include <autodiff/node.hpp>
#include <autodiff/activations.hpp>
#include <autodiff/operations.hpp>
#include <autodiff/computational_graph.hpp>
#include <autodiff/loss_fns.hpp>
#include <string>

extern std::unordered_map<std::string, std::function<Node*()>> ACTIVATION_FUNCTIONS;

extern std::unordered_map<std::string, std::function<LossNode*()>> LOSS_FUNCTIONS;

using namespace std;

// Represents a Fully Connected Segment between two layers of neurons
class FCSegment{

    public:

        Node* n1; // First layer of neurons
        Node* n2; // Second layer of neurons. This layer will be re-populated by FCSegment.

        ChildlessNode* bias; // Bias for each neuron in the second layer
        ChildlessNode* weights; // Weights between the two layers

        FCSegment(Node* n1, Node* n2, ComputationalGraph* graph);
};