#include <vector>
#include <functional>
#include <iostream>
#include "../units/units.hpp"

class Model {

public:
    
    Node* input;
    Node* output;
    Node* target;

    LossNode * loss;
    ComputationalGraph* graph;

    double learning_rate;

    Model(){
        this->loss = NULL;
    }

    double get_loss(){
        return loss->value;
    }

    virtual void forward() = 0;
    virtual void load_weights(string weights_path) = 0;
    virtual void backward() = 0;

    // Logging
    virtual void log_weights() = 0;
};

// Out-of-the-box models

class Perceptron : public Model
{
    public:
    
        Perceptron(string activation, string loss, int input_size, double learning_rate);
        void forward() override;
        void backward() override;
        void load_weights(string weights_path) override;
        void log_weights() override;

        FCSegment* s1;
};

struct MLPUnit
{
    int input_size;
    int output_size;
    string activation;
};

class MLP : public Model
{
    public:
    
        MLP(string loss, vector<MLPUnit> mlp_unit, double learning_rate);
        void forward() override;
        void backward() override;
        void load_weights(string weights_path) override;
        void log_weights() override;

        vector<FCSegment*> segments;

    private:
        vector<vector<Node*>> hidden_layers;
};