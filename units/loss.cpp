#include "../units/units.hpp"
#include <cmath>

using namespace std;

unordered_map<string, loss_function> LOSS_FUNCTIONS = {
    {"squared_error", squared_error}
};

double squared_error(vector<neuron> n1, vector<neuron> n2){
    double error = 0;
    for (int i = 0; i < n1.size(); i++){
        error += pow(n1[i].activation - n2[i].activation, 2);
    }
    return error;
}