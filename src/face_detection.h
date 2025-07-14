#ifndef FACEDETECTION_FACE_DETECTION_H
#define FACEDETECTION_FACE_DETECTION_H

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "../include/json.hpp"
#include "database.h"

using json = nlohmann::json;

const std::map<std::string, int> str2backend{
        {"opencv", cv::dnn::DNN_BACKEND_OPENCV}, {"cuda", cv::dnn::DNN_BACKEND_CUDA},
        {"timvx",  cv::dnn::DNN_BACKEND_TIMVX},  {"cann", cv::dnn::DNN_BACKEND_CANN}
};
const std::map<std::string, int> str2target{
        {"cpu", cv::dnn::DNN_TARGET_CPU}, {"cuda", cv::dnn::DNN_TARGET_CUDA},
        {"npu", cv::dnn::DNN_TARGET_NPU}, {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
};

cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces, const std::string& mode, Database& db, float fps = -1.f);

#endif //FACEDETECTION_FACE_DETECTION_H
