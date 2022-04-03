cmake_minimum_required(VERSION 2.8.15)
project(useGeometry)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# Add a header file
include_directories("/usr/local/include/eigen3")
add_executable(useGeometry useGeometry.cpp)
