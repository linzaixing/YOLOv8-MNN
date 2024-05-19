cmake_minimum_required(VERSION 3.5)

project(Yolov8MNN VERSION 0.1)

option(BUILD_YOLO_SHARED_LIBS "Build using shared libraries" OFF)
option(BUILD_WITH_MNN "Build the project with mnn" ON)
option(BUILD_SAMPLES "Build the Samples" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(YOLO_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

if(BUILD_WITH_MNN)
    add_definitions(-DBVA_USE_MNN)
endif()

if(BUILD_SAMPLES)
    message("Build project with samples")
    add_subdirectory(examples)
endif()

add_subdirectory(src)
