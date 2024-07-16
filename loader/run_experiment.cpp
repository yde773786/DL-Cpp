#include <iostream>
#include <libconfig.h++>
#include <fstream>
#include "../models/model.hpp"
#include "data_loader.hpp"

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif


using namespace libconfig;

Model* get_model_from_config(Setting& hyp_cfg, Setting& des_cfg,string model_type, string vectorization){
    if(model_type == "perceptron"){
        string activation_function, loss_function;
        int input_size;

        hyp_cfg.lookupValue("activation", activation_function);
        hyp_cfg.lookupValue("loss", loss_function);

        des_cfg.lookupValue("input_size", input_size);

        return new Perceptron(ACTIVATION_FUNCTIONS[activation_function], LOSS_FUNCTIONS[loss_function], input_size);
    }
    else{
        return NULL;
    }
}

pair<DataLoaderBase*, DataLoaderBase*> get_data_loader_from_config(Setting& cfg, string dataset_type){
    if(dataset_type == "playground"){
        string data_path;
        int batch_size;

        cfg.lookupValue("data_path", data_path);

        PlaygroundDataset* dataset = new PlaygroundDataset(data_path);
        cfg.lookupValue("data_path", data_path);

        double split_ratio = 0;
        cfg.lookupValue("split_ratio", split_ratio);

        cfg.lookupValue("batch_size", batch_size);

        vector<int> train_indices;
        vector<int> test_indices;

        LOG_DEBUG("Creating data loader with split ratio: %f", split_ratio);

        split_ratio = dataset->length * split_ratio;
        LOG_DEBUG("Train from: 0 to %f", split_ratio);

        for(int i = 0; i < dataset->length; i++){
            if(i < split_ratio){
                train_indices.push_back(i);
            }
            else{
                test_indices.push_back(i);
            }
        }

        LOG_DEBUG("Train indices size: %ld", train_indices.size());
        LOG_DEBUG("Test indices size: %ld", test_indices.size());

        return {new PlaygroundDataLoader(dataset, batch_size, train_indices) , new PlaygroundDataLoader(dataset, 1, test_indices)};
    }
    else{
        return {NULL, NULL};
    }
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
    int input_size;
    string weights_path = "";
    pair<DataLoaderBase*, DataLoaderBase*> data_loader_pair;
    DataLoaderBase* train_data_loader;
    DataLoaderBase* test_data_loader;
    Model* model;

    // Get the configurations
    try
    {
        model_type = cfg.lookup("model_type").c_str();
        vectorization = cfg.lookup("vectorization").c_str();
        dataset_type = cfg.lookup("dataset_type").c_str();

        data_loader_pair = get_data_loader_from_config(cfg.getRoot()["dataset"], dataset_type);
        train_data_loader = data_loader_pair.first;
        test_data_loader = data_loader_pair.second;

        if(!train_data_loader){
            cerr << "Data loader not found" << endl;
            return EXIT_FAILURE;
        }

        model = get_model_from_config(cfg.getRoot()["model_hyperparameters"], cfg.getRoot()["model_design"], model_type, vectorization);

        if(!model){
            cerr << "Model not found" << endl;
            return EXIT_FAILURE;
        }

        bool is_pre_trained;
        cfg.getRoot()["test"].lookupValue("is_pre_trained", is_pre_trained);
        if(is_pre_trained){
            cfg.getRoot()["test"].lookupValue("weights_path", weights_path);
        }
    }
    catch(const SettingNotFoundException &nfex)
    {
        cerr <<  "setting in configuration file." << endl;
    }

    if(weights_path == ""){
        // Train the model if not pre-trained
    }  
    else{
        // Load the weights
        LOG_DEBUG("Loading weights from %s", weights_path.c_str());
        model->load_weights(weights_path);
    }

    // Test the model
    if(dataset_type == "playground"){

        PlaygroundDataLoader * test_playground_data_loader = (PlaygroundDataLoader*) test_data_loader;

        LOG_DEBUG("Playground dataset length: %d", test_playground_data_loader->dataset->length);
        LOG_DEBUG("Number of batches: %d", test_playground_data_loader->dataset->length - test_playground_data_loader->batch_size);

        for(int i = 0; i < test_playground_data_loader->indices.size(); i++){
            auto batch = test_playground_data_loader->get_batch(i);
            for(int j = 0; j < batch.size(); j++){
                auto data = batch[j];
                model->input[0].activation = data.first.first;
                model->input[1].activation = data.first.second;

                LOG_DEBUG("Data: %f, %f", data.first.first, data.first.second);
                LOG_DEBUG("Label: %d", data.second);
                model->forward();
                vector<neuron> target = {{(double) data.second}};
                model->loss_and_predict(target);
            }
        }
    }

    return 0;
}