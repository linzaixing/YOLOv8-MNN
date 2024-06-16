
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

set(YOLO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(YOLO_INCLUDE_DIR ${YOLO_DIR}/include)

file(GLOB_RECURSE public_include_files
    ${YOLO_INCLUDE_DIR}/*.hh
    ${YOLO_INCLUDE_DIR}/*.h
    ${YOLO_INCLUDE_DIR}/*.hpp
    )

add_subdirectory(3party)

## set modules source dir
set(MODULES_SOURCE_DIR
    ${YOLO_SOURCE_DIR}
    )

set(yolo_include_dirs   
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YOLO_INCLUDE_DIR}
    )

## get all files
foreach(file_path ${MODULES_SOURCE_DIR})
    file(GLOB_RECURSE file_source 
            ${file_path}/*.cu ${file_path}/*.c
            ${file_path}/*.metal
            ${file_path}/*.cc ${file_path}/*.cpp 
            ${file_path}/*.mm ${file_path}/*.hh 
            ${file_path}/*.h ${file_path}/*.hpp
            ${file_path}/*.S
        )
    set(global_source ${global_source} ${file_source})
endforeach()

set(yolo_dependent_libs mnn ${OpenCV_LIBS})

message(STATUS "OpenCV_DIR=${OpenCV_DIR}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")

if(WIN32) 
    message(STATUS "platform: windows")
    set(windows_source ${global_source})
    set(yolo_source
        ${windows_source}
        ${public_include_files}
        )
elseif(LINUX)
    message(STATUS "platform: linux")
    set(linux_source ${global_source})
    set(yolo_source
        ${linux_source}
        )
elseif(ANDROID)
    message(STATUS "platform: android")
    set(android_source ${global_source})
    set(yolo_source
        ${android_source}
        )
endif()

if (MSVC)
    # compile static lib, surrpot Winwows
    add_library(yolov8 STATIC ${yolo_source})
else()
    # compile dynamic so, support Linux/Mac
    add_library(yolov8 SHARED ${yolo_source})
endif()

target_link_libraries(yolov8 PUBLIC ${yolo_dependent_libs})
target_include_directories(yolov8 PUBLIC 
    "$<BUILD_INTERFACE:${yolo_include_dirs}>"
    "$<INSTALL_INTERFACE:include>"
)
 