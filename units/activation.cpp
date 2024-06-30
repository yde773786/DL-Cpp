#include <cmath>
#include "units.hpp"

using namespace std;

double ReLU(double x){
    return x > 0 ? x : 0;
}

double sigmoid(double x){
    return 1 / (1 + exp(-x));
}