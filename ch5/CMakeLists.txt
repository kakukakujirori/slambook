cmake_minimum_required(VERSION 3.0)
project(ch5)

set(CMAKE_BUILD_TYPE "Release")
# use c++11 standard
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# Eigen
include_directories("/usr/include/eigen3")

# find OpenCV
find_package(OpenCV REQUIRED)
# include its headers
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
add_subdirectory(stereo)
add_subdirectory(rgbd)
