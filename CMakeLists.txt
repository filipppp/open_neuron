cmake_minimum_required (VERSION 3.8)

add_subdirectory(arrayhelper)
add_subdirectory(neuralmath)
add_subdirectory(network)


add_executable(main main.cpp main.h)
set(INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dist/")

target_link_libraries(main PUBLIC neuralmath arrayhelper network)
