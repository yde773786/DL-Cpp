cmake_minimum_required(VERSION 3.10)

# Project name and version
project(DL-CPP VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(models units units/autodiff)

# Add the libconfig library
find_library(LIBCONFIG libconfig++.so)

# Add the macro-logger library
add_subdirectory(macro-logger)

# Add the autodiff library
add_subdirectory(units/autodiff)

# Specify the source files for the executable
set(SOURCES
    models/model.hpp
    models/perceptron.cpp
    units/units.cpp
    units/units.hpp
    loader/data_loader.hpp
    loader/data_loader.cpp
    loader/run_experiment.cpp
)

# Add the executable
add_executable(dl_cpp ${SOURCES})

# Link the libconfig library
target_link_libraries(dl_cpp macro-logger ${LIBCONFIG})

# Link the autodiff library
target_link_libraries(dl_cpp autodiff)

target_compile_definitions(dl_cpp PRIVATE LOG_LEVEL=$ENV{LOG_LEVEL})