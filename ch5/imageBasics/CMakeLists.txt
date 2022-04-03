project(imageBasics)

add_executable(imageBasics imageBasics.cpp)
# link OpenCV
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
# link OpenCV
target_link_libraries(undistortImage ${OpenCV_LIBS})
