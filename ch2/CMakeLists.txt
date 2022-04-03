cmake_minimum_required( VERSION 2.8.12 )
project( HelloSLAM )

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
set(CMAKE_MACOSX_RPATH 1)
endif()

Set( CMAKE_BUILD_TYPE "Debug" )

add_library( hello_shared SHARED libHelloSLAM.cpp )
add_executable( helloSLAM helloSLAM.cpp )
add_executable( useHello useHello.cpp )
target_link_libraries( useHello hello_shared )