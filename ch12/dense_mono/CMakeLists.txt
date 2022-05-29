cmake_minimum_required(VERSION 3.0)
project(dense_monocular)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

############### dependencies ######################
# Eigen
include_directories("/usr/local/include/eigen3")
# OpenCV
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(dense_mapping dense_mapping.cpp)
target_link_libraries(dense_mapping ${OpenCV_LIBS} Sophus::Sophus)