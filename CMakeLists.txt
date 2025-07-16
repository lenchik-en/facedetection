cmake_minimum_required(VERSION 3.10)
project(facedetection)

set(CMAKE_CXX_STANDARD 17)

# === OpenCV ===
set(OpenCV_DIR "/opt/opencv-min/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# === Исходники ===
add_executable(main
        src/main.cpp
        src/visualiser.cpp
        src/database.h
        src/face_detection.h
        src/YuNet.h
)

target_include_directories(main PRIVATE src)

target_link_libraries(main ${OpenCV_LIBS})
