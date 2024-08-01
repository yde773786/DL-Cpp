add_library(autodiff OBJECT
    units.hpp
    units.cpp
    computational_graph.hpp
    computational_graph.cpp
    tests/test_autodiff.cpp
    )

add_executable(test_autodiff $<TARGET_OBJECTS:autodiff>)

target_include_directories(autodiff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})