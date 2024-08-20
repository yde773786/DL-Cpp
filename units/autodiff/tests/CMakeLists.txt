set(CMAKE_BUILD_TYPE Debug)
add_library(test_autodiff OBJECT
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

# Add the macro-logger library
add_subdirectory("../../../macro-logger" "${CMAKE_CURRENT_BINARY_DIR}/macro_logger")

# Link the libconfig library
target_link_libraries(test_autodiff macro-logger ${LIBCONFIG})

add_executable(test_autodiff_exe $<TARGET_OBJECTS:test_autodiff>)

target_include_directories(test_autodiff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})