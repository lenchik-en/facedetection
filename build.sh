#!/bin/bash

set -e

rm -rf build
mkdir -p build && cd build

cmake ..
make

./main --model=/Users/elenagrigoreva/CLionProjects/facedetection/face_detection_yunet_2023mar.onnx

