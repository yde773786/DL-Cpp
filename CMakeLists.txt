cmake_minimum_required(VERSION 3.10)

# Project name and version
project(NeuralNetwork VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(models units units/autodiff)

# Specify the source files for the executable
set(SOURCES
    models/model.hpp
    models/perceptron.cpp
    models/run_experiment.cpp
    units/activation.hpp
    units/loss.hpp
    units/units.cpp
    units/units.hpp
)

# Add the executable
add_executable(dl_cpp ${SOURCES})
