cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

set(PROJECT_SOURCES
    detect_demo.cc
)

add_executable(yolov8_detect ${PROJECT_SOURCES})
target_link_libraries(yolov8_detect yolov8)

install(TARGETS yolov8_detect RUNTIME DESTINATION bin)
