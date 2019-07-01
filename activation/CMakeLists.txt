cmake_minimum_required (VERSION 3.8)

add_library(functions functions.cpp functions.h)
target_include_directories(functions PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

