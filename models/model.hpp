#include <vector>
#include <iostream>
#include "../units/units.hpp"

class Model {

public:
    
    vector<Node*> input;
    vector<Node*> output;
    vector<Node*> target;

    Node* loss;
    ComputationalGraph* graph;

    Model(Node* loss) : loss(loss) {};
    
    double get_loss(){
        return loss->value;
    }

    virtual void forward() = 0;
    virtual void load_weights(string weights_path) = 0;
    virtual void backward(vector<double> target_val) = 0;
};

// Out-of-the-box models

class Perceptron : public Model
{
    public:
    
        Perceptron(Node* activation, Node* loss, int input_size);
        void forward() override;
        void backward(vector<double> target_val) override;
        void load_weights(string weights_path) override;

        FCSegment* s1;
};