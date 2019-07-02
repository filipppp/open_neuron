cmake_minimum_required (VERSION 3.8)

add_library(arrayhelper src/arrayhelper.cpp src/arrayhelper.h)
target_include_directories(arrayhelper PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(arrayhelper PUBLIC network neuralmath)



