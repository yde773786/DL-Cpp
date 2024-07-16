#include <cmath>
#include "units.hpp"

using namespace std;

unordered_map<string, activation_function> ACTIVATION_FUNCTIONS = {
    {"sigmoid", sigmoid},
    {"ReLU", ReLU},
    {"tanh", tanh}
};

double ReLU(double x){
    return x > 0 ? x : 0;
}

double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

double tanh(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}