#include <vector>

using namespace std;

struct neuron
{
    double activation;
};

struct weight
{
    double value;
};

class Segment
{
    public:

        vector<neuron> n1;
        vector<neuron> n2;
        vector<vector<weight>> weights;

        Segment(vector<neuron> n1, vector<neuron> n2){
            this->n1 = n1;
            this->n2 = n2;

            for (int i = 0; i < n1.size(); i++){
                vector<weight> w;
                for (int j = 0; j < n2.size(); j++){
                    weight new_weight;
                    new_weight.value = 0;
                    w.push_back(new_weight);
                }
                weights.push_back(w);
            }
        }
};