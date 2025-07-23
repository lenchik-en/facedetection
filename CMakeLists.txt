cmake_minimum_required(VERSION 3.10)
project(facedetection)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
set(OpenCV_DIR "/opt/opencv-min/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# Исходники
set(SOURCES
        src/main.cpp
        src/visualiser.cpp
        src/database.cpp
        src/YuNet.cpp
)
add_executable(main ${SOURCES})

target_include_directories(main PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})

# Google Test
add_subdirectory(include/googletest)
enable_testing()

file(GLOB TEST_SOURCES "src/tests/*.cpp")
add_executable(test_database ${TEST_SOURCES})
target_include_directories(test_database PRIVATE src)
target_link_libraries(test_database gtest gtest_main ${OpenCV_LIBS})

add_test(NAME test_database COMMAND test_database)
