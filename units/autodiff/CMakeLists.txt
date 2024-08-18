# Add the libconfig library
find_library(LIBCONFIG libconfig++.so)

add_library(autodiff OBJECT
    node.hpp
    computational_graph.hpp
    computational_graph.cpp
    loss_fns.hpp
    loss_fns.cpp
    operations.hpp
    operations.cpp
    activations.hpp
    activations.cpp
    )

# Link the libconfig library
target_link_libraries(autodiff macro-logger ${LIBCONFIG})

target_include_directories(autodiff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})