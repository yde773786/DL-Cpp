#include <vector>
#include <iostream>
#include "../units/units.hpp"

class Model {

public:
    
    vector<neuron> input;
    vector<neuron> output;

    activation_function activation;
    loss_function loss;

    Model(activation_function activation, loss_function loss) : activation(activation), loss(loss) {};
    
    void loss_and_predict(vector<neuron> target){
        int max = 0;
        for(int i = 0; i < output.size(); i++){
            if(output[i].activation > max){
                max = i;
            }
        }

        cout << "Predicted value: " << max << endl;
        cout << "Loss value" << loss(output, target) << endl;
    }
};

class Perceptron : public Model
{
    public:
    
        neuron middle;

        Perceptron(activation_function activation, loss_function loss, int input_size);
        void forward();
};