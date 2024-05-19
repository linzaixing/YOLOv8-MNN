
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
    ${YOLO_SOURCE_DIR}/util
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

foreach(file ${global_source})
    # split mnn files
    string(REGEX MATCH "${YOLO_DIR}/.*/mnn/.*" mnn_files ${file})
    if (mnn_files)
        set(mnn_source ${mnn_source} ${mnn_files})
    endif()
endforeach()

if(NOT BUILD_WITH_MNN)
    list(REMOVE_ITEM global_source ${mnn_source})
endif()

set(yolo_dependent_libs mnn ${OpenCV_LIBS})

message(STATUS "OpenCV_DIR=${OpenCV_DIR}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")

## compile options
set(compile_options "")

if(LINUX)
    list(APPEND compile_options PRIVATE -Wall -Wextra)
endif()

## build target bva
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
