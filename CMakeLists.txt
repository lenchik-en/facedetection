cmake_minimum_required(VERSION 3.10)
project(facedetection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(libfacedetection/src)
include_directories(libfacedetection/build)

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libfacedetection/build/libfacedetection.a)
