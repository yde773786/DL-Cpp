#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

using namespace std;

// A wrapper class for data loaders, for readability when data loading
class DataLoaderBase {
    public:
        unordered_map<string, float> label_to_value;
        unordered_map<float, string> value_to_label;

        DataLoaderBase() = default;
};

template <typename T>
class DataLoader : public DataLoaderBase{

    public:
        DataLoader(string& label_path);

        vector<pair<vector<T>, vector<int>>>  data;

        // Pair of data vector and label vector
        vector<pair<vector<T>, vector<int>>> test_data;
        vector<pair<vector<T>, vector<int>>> train_data;
        vector<pair<vector<T>, vector<int>>> val_data;


        // Returns a batch of pairs of data vector and label vector. Each getter is without replacement.
        virtual pair<vector<T>, vector<int>> get_test_data() = 0;
        virtual vector<pair<vector<T>, vector<int>>> get_train_data(int batch_size) = 0;
        virtual vector<pair<vector<T>, vector<int>>> get_val_data(int batch_size) = 0;
};

class FeatureDataLoader : public DataLoader<float> {
    public:
        FeatureDataLoader(string& label_path, string& data_path);

        pair<vector<float>, vector<int>> get_test_data() override;
        vector<pair<vector<float>, vector<int>>> get_train_data(int batch_size) override;
        vector<pair<vector<float>, vector<int>>> get_val_data(int batch_size) override;
};

