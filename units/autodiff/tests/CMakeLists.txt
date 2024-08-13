add_library(autodiff OBJECT
    ../node.hpp
    ../computational_graph.hpp
    ../computational_graph.cpp
    test_autodiff.cpp
    ../loss_fns.hpp
    ../loss_fns.cpp
    ../operations.hpp
    ../operations.cpp
    ../activations.hpp
    ../activations.cpp
    )

add_executable(test_autodiff $<TARGET_OBJECTS:autodiff>)

target_include_directories(autodiff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})