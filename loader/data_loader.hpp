#include <string>
#include <unordered_map>
#include <vector>
#include "../models/model.hpp"
#include <utility>

using namespace std;

// This is a base class for all Datasets
template <typename T, typename U>
class Dataset{
    public:
        int length;
        vector<pair<T, U>> data_to_label;
        virtual pair<T, U> get_data(int index) = 0;
};

// Define your custom dataset here

class PlaygroundDataset : public Dataset<pair<float, float>, int>{

    public:
        PlaygroundDataset(string& data_path);
        pair<pair<float, float>, int> get_data(int index) override;
};

// This is a base class for all DataLoaders. Serves as a wrapper for the Dataset class
class DataLoaderBase{
    public:
        int batch_size;
};

// This is a base class for all DataLoaders

template <typename T, typename U>
class DataLoader: public DataLoaderBase{
    public:
        vector<int> indices;
        Dataset<T, U>* dataset;

        virtual vector<pair<T,U>> get_batch(int index) = 0;
        virtual double test(Model* model) = 0;
};

// Define your custom DataLoader here

class PlaygroundDataLoader : public DataLoader<pair<float, float>, int>{
    public:
        PlaygroundDataLoader(PlaygroundDataset* dataset, int batch_size, vector<int> indices);
        vector<pair<pair<float, float>, int>> get_batch(int index) override;
        double test(Model* model) override;
};