cmake_minimum_required(VERSION 3.16)
project(facedetection)

set(CMAKE_CXX_STANDARD 17)

#OpenCV options
set(OPENCV_EXTRA_MODULES_PATH ${CMAKE_SOURCE_DIR}/include/opencv_contrib/modules)
set(BUILD_LIST core,imgproc,dnn,videoio,highgui,imgcodecs,objdetect,face)
set(BUILD_opencv_apps OFF)
set(BUILD_DOCS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_TESTS OFF)
set(BUILD_PERF_TESTS OFF)
set(BUILD_opencv_world OFF)
set(WITH_CUDA OFF)

#Собираем OpenCV
add_subdirectory(include/opencv)

#Исходники
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_executable(main ${SOURCES})
target_include_directories(main PRIVATE
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/opencv2 # для opencv_modules.hpp
        ${OpenCV_SOURCE_DIR}/modules/core/include
        ${OpenCV_SOURCE_DIR}/modules/imgproc/include
        ${OpenCV_SOURCE_DIR}/modules/highgui/include
        ${OpenCV_SOURCE_DIR}/modules/videoio/include
        ${OpenCV_SOURCE_DIR}/modules/imgcodecs/include
        ${OpenCV_SOURCE_DIR}/modules/dnn/include
        ${OpenCV_SOURCE_DIR}/modules/objdetect/include
        ${OpenCV_SOURCE_DIR}/modules/calib3d/include
        ${OpenCV_SOURCE_DIR}/modules/features2d/include
        ${OpenCV_SOURCE_DIR}/modules/flann/include
        ${OpenCV_SOURCE_DIR}/modules/photo/include
        ${OPENCV_EXTRA_MODULES_PATH}/face/include
        ${OpenCV_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/googletest/googletest/include
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(main
        opencv_core
        opencv_imgproc
        opencv_dnn
        opencv_highgui
        opencv_videoio
        opencv_imgcodecs
        opencv_objdetect
        opencv_face
        gtest
        gtest_main
)

add_dependencies(main opencv_core)


# Google Test
enable_testing()
add_subdirectory(include/googletest)

file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(test_database
        ${TEST_SOURCES}
        ${CMAKE_SOURCE_DIR}/src/visualiser.cpp)
target_include_directories(test_database PRIVATE
        ${CMAKE_SOURCE_DIR}/include/googletest/googletest/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/opencv2
        ${OpenCV_SOURCE_DIR}/modules/core/include
        ${OpenCV_SOURCE_DIR}/modules/imgproc/include
        ${OpenCV_SOURCE_DIR}/modules/highgui/include
        ${OpenCV_SOURCE_DIR}/modules/videoio/include
        ${OpenCV_SOURCE_DIR}/modules/imgcodecs/include
        ${OpenCV_SOURCE_DIR}/modules/dnn/include
        ${OpenCV_SOURCE_DIR}/modules/objdetect/include
        ${OpenCV_SOURCE_DIR}/modules/calib3d/include
        ${OpenCV_SOURCE_DIR}/modules/features2d/include
        ${OpenCV_SOURCE_DIR}/modules/flann/include
        ${OpenCV_SOURCE_DIR}/modules/photo/include
        ${OPENCV_EXTRA_MODULES_PATH}/face/include
        ${OpenCV_SOURCE_DIR}/include
)
target_link_libraries(test_database
    gtest
    gtest_main
    opencv_core
    opencv_imgproc
    opencv_dnn
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_objdetect
    opencv_face
)
add_test(NAME test_database COMMAND test_database)
