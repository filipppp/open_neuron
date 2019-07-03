cmake_minimum_required (VERSION 3.8)

add_library(
    core
    src/core/functions.cpp
    src/core/functions.h
    src/core/matrix.cpp
    src/core/matrix.h
    src/core/network.cpp
    src/core/network.h
    src/core/layer.cpp
    src/core/layer.h
    src/core/arrayhelper.cpp
    src/core/arrayhelper.h
    )
target_include_directories(core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")


