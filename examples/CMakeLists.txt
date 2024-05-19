cmake_minimum_required(VERSION 3.5)

option(BUILD_SAMPLE_DETECT OFF "yolov8 detect sample")

if (BUILD_SAMPLE_DETECT)
    add_subdirectory(detect)
endif ()