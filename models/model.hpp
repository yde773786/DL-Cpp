#include <vector>
#include <iostream>
#include "../units/units.hpp"

class Model {

public:
    
    vector<neuron> input;
    vector<neuron> output;

    loss_function loss;

    Model(loss_function loss) : loss(loss) {};
    
    double get_loss(vector<neuron> target){
        int max = 0;
        for(int i = 0; i < output.size(); i++){
            if(output[i].activation > max){
                max = i;
            }
        }

        return loss(target, output);
    }

    virtual void forward() = 0;
    virtual void load_weights(string weights_path) = 0;
};

class Perceptron : public Model
{
    public:
    
        Perceptron(activation_function activation, loss_function loss, int input_size);
        void forward() override;
        void load_weights(string weights_path) override;

        Segment* s1;
};