#include "data_loader.hpp"
#include <fstream>
#include <sstream>

#ifndef MACROLOGGER_H
#define MACROLOGGER_H
#include <macrologger.h>
#endif

using namespace std;

// Define your custom dataset here

PlaygroundDataset::PlaygroundDataset(string& data_path){
    
    // Load the data from the file
    ifstream data_file(data_path);
    string line;
    int line_count = 0;
    int label;

    while(getline(data_file, line)){

        line_count++;
        pair<float, float> data;
        int label;
        stringstream ss(line);

        LOG_DEBUG("Line: %s", line.c_str());

        int counter = 0;

        for(float i; ss >> i;){
            if(counter == 0){
                LOG_DEBUG("X: %f", i);
                data.first = i;
            }
            else if(counter == 1){
                LOG_DEBUG("Y: %f", i);
                data.second = i;
            }
            else{
                LOG_DEBUG("Label: %d", (int) i);
                label = (int) i;
            }

            if(ss.peek() == ','){
                ss.ignore();
                counter++;
            }
        }

        data_to_label.push_back(make_pair(data, label));
    }

    length = line_count;
    LOG_DEBUG("Length: %d", length);
}

pair<pair<float, float>, int> PlaygroundDataset::get_data(int index){
    LOG_DEBUG("Getting data at index: %d", index);
    return data_to_label[index];
}

// Define your custom DataLoader here

PlaygroundDataLoader::PlaygroundDataLoader(PlaygroundDataset* dataset, int batch_size, vector<int> indices){
    this->dataset = dataset;
    this->batch_size = batch_size;
    this->indices = indices;
}

vector<pair<pair<float, float>, int>> PlaygroundDataLoader::get_batch(int index){
    vector<pair<pair<float, float>, int>> batch;

    int actual_index = index * batch_size;

    for(int i = actual_index; i < actual_index + batch_size; i++){
        LOG_DEBUG("Getting data at index: %d", indices[i]);
        batch.push_back(dataset->get_data(indices[i]));
    }

    return batch;
}

double PlaygroundDataLoader::test(Model* model){
    double correct = 0;
    double total = 0;

    for(int i = 0; i < indices.size(); i++){
        vector<pair<pair<float, float>, int>> batch = get_batch(i);

        for(int j = 0; j < batch.size(); j++){
            auto data = batch[j];

            model->input[0]->value = data.first.first;
            model->input[1]->value = data.first.second;

            LOG_DEBUG("Data: %f, %f", data.first.first, data.first.second);
            LOG_DEBUG("Label: %d", data.second);

            model->target[0]->value = data.second;
            model->forward();

            cout << "Loss: " << model->get_loss() << endl;

            int predicted = model->output[0]->value > 0 ? 1 : -1;

            LOG_DEBUG("Activation: %f", model->output[0]->value);
            LOG_DEBUG("Predicted: %d", predicted);
            LOG_DEBUG("Correct: %d", data.second);
            if(predicted == data.second){
                correct++;
            }
            total++;
        }
    }

    return correct / total;
}

double PlaygroundDataLoader::train(Model* model){
    double correct = 0;
    double total = 0;

    for(int i = 0; i < indices.size(); i++){
        vector<pair<pair<float, float>, int>> batch = get_batch(i);

        for(int j = 0; j < batch.size(); j++){
            auto data = batch[j];

            model->input[0]->value = data.first.first;
            model->input[1]->value = data.first.second;

            LOG_DEBUG("Data: %f, %f", data.first.first, data.first.second);
            LOG_DEBUG("Label: %d", data.second);

            model->target[0]->value = data.second;
            model->forward();

            cout << "Loss: " << model->get_loss() << endl;

            int predicted = model->output[0]->value > 0 ? 1 : -1;

            LOG_DEBUG("Activation: %f", model->output[0]->value);
            LOG_DEBUG("Predicted: %d", predicted);
            LOG_DEBUG("Correct: %d", data.second);

            if(predicted == data.second){
                correct++;
            }
            total++;

            model->backward();
            model->graph->apply_grad(model->learning_rate);
        }
    }

    return correct / total;
}