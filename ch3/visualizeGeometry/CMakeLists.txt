cmake_minimum_required(VERSION 2.8.15)
project(visualizeGeometry)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# Add a header file
include_directories("/usr/local/include/eigen3")
find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(visualizeGeometry visualizeGeometry.cpp)
target_link_libraries(visualizeGeometry ${Pangolin_LIBRARIES})
