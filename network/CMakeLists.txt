cmake_minimum_required (VERSION 3.8)

add_library(
    network
    src/layer.cpp
    src/layer.h
    src/network.cpp
    src/network.h
)
target_include_directories(network PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(network PUBLIC arrayhelper neuralmath)
