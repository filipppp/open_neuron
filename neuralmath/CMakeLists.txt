cmake_minimum_required (VERSION 3.8)

add_library(
    neuralmath
    src/functions.cpp
    src/functions.h
    src/matrix.cpp
    src/matrix.h
    )
target_include_directories(neuralmath PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(neuralmath PUBLIC arrayhelper)


