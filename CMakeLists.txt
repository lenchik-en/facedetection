cmake_minimum_required(VERSION 3.10)
project(facedetection)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
set(OpenCV_DIR "/opt/opencv-4.13/lib/cmake/opencv4")
find_package(OpenCV 4.5.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main src/main.cpp
        src/face_detection.h
        src/YuNet.h
        src/visualiser.cpp
        src/database.h)
target_link_libraries(main ${OpenCV_LIBS})