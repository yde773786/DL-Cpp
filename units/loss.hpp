#include "../units/units.cpp"
#include <cmath>

using namespace std;

inline double squaredError(vector<neuron> n1, vector<neuron> n2){
    double error = 0;
    for (int i = 0; i < n1.size(); i++){
        error += pow(n1[i].activation - n2[i].activation, 2);
    }
    return error;
}