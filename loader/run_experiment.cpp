#include <iostream>
#include <libconfig.h++>
#include <fstream>
#include "../models/model.hpp"
#include "data_loader.hpp"

using namespace libconfig;

template <typename T>
DataLoader<T>::DataLoader(std::string& label_path) : DataLoaderBase(){
    ifstream label_file(label_path);

    int count = 0;

    string line;
    while(getline(label_file, line)){
        count++;
        label_to_value[line] = count;
        value_to_label[count] = line;
    }
}

FeatureDataLoader::FeatureDataLoader(string& label_path, string& data_path) : DataLoader<float>(label_path){
    ifstream data_file(data_path);

    string line;
    while(getline(data_file, line)){
        vector<float> data_vector;
        vector<int> label_vector;

        size_t pos = 0;
        string token;
        while ((pos = line.find(',')) != string::npos) {
            token = line.substr(0, pos);
            data_vector.push_back(stof(token));
            line.erase(0, pos + 1);
        }
        label_vector = vector<int>(1, 0);
        label_vector[label_to_value[line]] = 1;

        data.push_back({data_vector, label_vector});
    }
}

vector<pair<vector<float>, vector<int>>> FeatureDataLoader::get_test_data(int batch_size){
    return test_data;
}

vector<pair<vector<float>, vector<int>>> FeatureDataLoader::get_train_data(int batch_size){
    return train_data;
}

vector<pair<vector<float>, vector<int>>> FeatureDataLoader::get_val_data(int batch_size){
    return val_data;
}

Model* get_model_from_config(Config& cfg, string model_type, string vectorization){
    return new Model(ReLU, squared_error);
}

DataLoaderBase* get_data_loader_from_config(Config& cfg, string dataset_type){
    return new DataLoaderBase();
}

int main(int argc, char **argv){
    std::cout << "DL-CPP" << std::endl;

    if(argc != 2){
        std::cout << "Usage: ./run_experiment <config_file>" << std::endl;
        return EXIT_FAILURE;
    }
    else{
        std::cout << "Config file: " << argv[1] << std::endl;
    }

    std::cout << "Loading configuration file" << std::endl;
    Config cfg;

    // As in libconfig example
    try{
        cfg.readFile(argv[1]);
    }
    catch(const FileIOException &fioex)
    {
        std::cerr << "I/O error while reading file." << std::endl;
        return(EXIT_FAILURE);
    }
    catch(const ParseException &pex)
    {
        std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                << " - " << pex.getError() << std::endl;
        return(EXIT_FAILURE);
    }

    string model_type, vectorization, dataset_type;

    // Get the configurations
    try
    {
        model_type = cfg.lookup("model_type").c_str();
        vectorization = cfg.lookup("vectorization").c_str();
        dataset_type = cfg.lookup("dataset_type").c_str();

        Model* model = get_model_from_config(cfg, model_type, vectorization);
        DataLoaderBase* data_loader = get_data_loader_from_config(cfg, dataset_type);
    }
    catch(const SettingNotFoundException &nfex)
    {
        cerr <<  "setting in configuration file." << endl;
    }

    return 0;
}