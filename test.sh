#!/bin/bash

set -e

rm -rf build
mkdir -p build && cd build

cmake ..
make

#./main --database=/Users/elenagrigoreva/CLionProjects/facedetection/data/face_db.json --model=/Users/elenagrigoreva/CLionProjects/facedetection/models/face_detection_yunet_2023mar.onnx --vm=/Users/elenagrigoreva/CLionProjects/facedetection/models/MobileFaceNet.onnx --mode=identify

./test_database
