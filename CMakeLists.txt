cmake_minimum_required(VERSION 3.10)

# Project name and version
project(NeuralNetwork VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(models units units/autodiff)

# Add the libconfig library
find_library(LIBCONFIG libconfig++.so)

# Specify the source files for the executable
set(SOURCES
    models/model.hpp
    models/perceptron.cpp
    units/units.cpp
    units/units.hpp
    units/activation.cpp
    units/loss.cpp
    loader/data_loader.hpp
    loader/run_experiment.cpp
)

# Add the executable
add_executable(dl_cpp ${SOURCES})

# Link the libconfig library
target_link_libraries(dl_cpp ${LIBCONFIG})