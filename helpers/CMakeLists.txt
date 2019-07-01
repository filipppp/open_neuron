cmake_minimum_required (VERSION 3.8)

add_library(arrayhelper arrayhelper.cpp arrayhelper.h)
target_include_directories(arrayhelper PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

