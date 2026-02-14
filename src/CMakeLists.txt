
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(YOLO_INCLUDE_DIR ${YOLO_DIR}/include)

file(GLOB_RECURSE public_include_files
    ${YOLO_INCLUDE_DIR}/*.hh
    ${YOLO_INCLUDE_DIR}/*.h
    ${YOLO_INCLUDE_DIR}/*.hpp
    )

add_subdirectory(3party)

## set modules source dir
set(MODULES_SOURCE_DIR
    ${SOURCE_DIR}
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
    set(file_source
        ${windows_source}
        ${public_include_files}
        )
elseif(UNIX AND NOT APPLE)
    # LINUX var exists only in CMake 3.25+, use UNIX for compatibility
    message(STATUS "platform: linux")
    set(linux_source ${global_source})
    set(file_source
        ${linux_source}
        )
elseif(ANDROID)
    message(STATUS "platform: android")
    set(android_source ${global_source})
    set(file_source
        ${android_source}
        )
endif()

if (MSVC)
    # compile static lib, surrpot Winwows
    add_library(yolov8 STATIC ${file_source})
else()
    # compile dynamic so, support Linux/Mac
    add_library(yolov8 SHARED ${file_source})
endif()

target_link_libraries(yolov8 PUBLIC ${yolo_dependent_libs})
target_include_directories(yolov8 PUBLIC 
    "$<BUILD_INTERFACE:${yolo_include_dirs}>"
    "$<INSTALL_INTERFACE:include>"
)

# install 时安装 yolov8 库（.so / .dll / .a）
install(TARGETS yolov8
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装头文件，便于其他项目链接使用
install(FILES ${public_include_files} DESTINATION include)
