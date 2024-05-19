cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

set(PROJECT_SOURCES
    detect_demo.cc
)

add_executable(yolov8_detect ${PROJECT_SOURCES})
target_link_libraries(yolov8_detect ${OpenCV_LIBS} mnn)

add_executable(image_fusion image_fusion.cpp)
target_link_libraries(image_fusion ${OpenCV_LIBS} mnn)

